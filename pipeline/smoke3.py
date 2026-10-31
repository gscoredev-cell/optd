"""SMOKE-3 (diagnostic, one pod, <= 90 min). Reuses SMOKE-2 artifacts from the volume (config smoke3_source_run).

(a) probe-mc on SMOKE-2's 300 greedy train traces: at the prompt and at every boundary, 3 sampled completions
    (T = 0.8, total cap 800 tokens); MC success = share correct under the primary rule. Step label = 1 if MC success
    <= 1/3 and the previous point's >= 2/3. Step probe with the grouped-CV protocol.
(b) Qwen3-4B student: S1 student-only on SMOKE-2's test set (teacher numbers reused), S2 on 300 GSM train prompts
    (100 each main/p1/p2) + 300 RandomCalculation (3 ops) x 2, S3 with the same protocol.
(c) Data scaling for the 1.7B: S2 on 900 GSM (300 per set) + 300 RC (3 ops) x 2 = 2,400 traces; S3 curve.

tau is calibrated on held-out negatives reweighted to the eval's set mix; eval FPR per set uses the test traces.
Each stage has a time budget: it is skipped when the pod's remaining time is shorter, and generation stops starting
new batches at the stage deadline (results then cover fewer traces, reported as such).
"""
import json
import os
import pathlib
import random
import statistics
import time

import torch

import data as D
import probe as P
import smoke
from core import holdout_by_oid, primary_ok, seed_for
from generate import decode, decode_batches, load_tokenizer, prompt_ids

MAX_NEW = 800                         # SMOKE-2's final max_new_tokens
MC_K, MC_GROUP = 3, 24                # completions per point, traces per scheduling group
EVAL_SHARE = {"p2": 200 / 380, "main": 40 / 380, "p1": 40 / 380, "rc": 100 / 380}
RC_TRAIN_SEED, RC_TRAIN_N = 200, 300
BUDGET_MIN = {"MCa": 18, "MCprobe": 5, "B1": 7, "B2": 12, "B3": 9, "C2": 12, "C3": 11}   # sum 74 of ~78 usable
B3_C_GRID = (1e-3, 1e-2, 1e-1)        # 4B features are 2.5x the work per trace; SMOKE-1/2 optima were 1e-3..1e-2
SKIP = "skipped: not enough pod time left"


def pipeline_deadline():
    """Epoch by which run_phase must have written the report: the pipeline timeout (counted from this container's
    bootstrap) minus a 2-minute margin, and never later than the pod's absolute deadline minus 2 minutes."""
    by_timeout = int(os.environ["BOOT_EPOCH"]) + int(os.environ["MAX_MINUTES"]) * 60 - 600 - 120
    return min(by_timeout, int(os.environ["DEADLINE_EPOCH"]) - 120)


class Ctx3:
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        self.run_id, self.run_dir = run_id, run_dir
        self.src = pathlib.Path("/workspace/runs") / cfg["smoke3_source_run"]
        self.s = smoke.Ctx(cfg, tiers, tier, run_id, run_dir)                         # Qwen3-1.7B
        self.b = smoke.Ctx({**cfg, "student": cfg["student_alt"]}, tiers, tier, run_id, run_dir)   # Qwen3-4B
        for c in (self.s, self.b):
            c.gen_bs *= 2
            c.max_new = lambda: MAX_NEW
        gsm = D.load_gsm(self._snapshot(cfg))
        train_oids, _ = D.make_split(gsm)
        rc = D.random_calculation(RC_TRAIN_N, seed=RC_TRAIN_SEED, min_ops=cfg["rc_ops_full"],
                                  max_ops=cfg["rc_ops_full"])
        pool = {c: D.problems(gsm, [c], train_oids) for c in ("main", "p1", "p2")}
        # fixed shuffle (not pid order): a time cut then drops problems from every set, not the last set (rc)
        self.train_b = random.Random(1).sample([p for c in pool for p in D.sample(pool[c], 100)] + rc, 600)
        self.train_c = random.Random(2).sample([p for c in pool for p in D.sample(pool[c], 300)] + rc, 1200)
        for c in (self.s, self.b):
            c.gold.update({p["pid"]: p["gold"] for p in self.train_c + self.train_b})
        self.group = {p["pid"]: (f"gsm:{p['oid']}" if p["oid"] is not None else p["pid"])
                      for p in self.train_c + self.train_b + self.s.train}

    @staticmethod
    def _snapshot(cfg):
        from huggingface_hub import snapshot_download
        return snapshot_download(cfg["dataset"], repo_type="dataset", local_files_only=True)

    def src_load(self, stage, name):
        return torch.load(self.src / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def budget(self, name):
        """(start allowed, stage deadline)."""
        mins = BUDGET_MIN[name]
        return time.time() + mins * 60 <= pipeline_deadline(), min(time.time() + mins * 60, pipeline_deadline())


def _rate(traces, key, s=None):
    xs = [t[key] for t in traces if s is None or t["set"] == s]
    return sum(xs) / len(xs) if xs else float("nan")


def _sets(traces):
    return [t["set"] for t in traces]


def _eval_fpr(ctx, model_probe, traces, flag_fn):
    """Per-set FPR on test traces: share of primary-correct traces the probe flags."""
    feats = ctx.feats(traces, model_probe["layers_all"])
    flagged = [bool(flag_fn(f)) for f in feats]
    out = {}
    for s in ("p2", "main", "p1", "rc", None):
        idx = [i for i, t in enumerate(traces) if t["ok"] and (s is None or t["set"] == s)]
        out[s or "pooled"] = {"n_correct": len(idx), "fpr": sum(flagged[i] for i in idx) / len(idx) if idx else None}
    return out


def mc_success(ctx, traces, run_id, tag, deadline, k=MC_K):
    """MC success at the prompt and at every boundary of each trace: share of k sampled completions (T 0.8, total cap
    MAX_NEW) correct under the primary rule. Whole groups of MC_GROUP traces only; no new group after the deadline.
    Returns ({pid: [mc at prompt, boundary 1, ...]}, completions, generated tokens)."""
    model = ctx.student()
    done, n_rows, n_tok = {}, 0, 0
    for g0 in range(0, len(traces), MC_GROUP):
        group = traces[g0:g0 + MC_GROUP]
        rows = [(t, j, cut, r) for t in group for j, cut in enumerate([0] + [b + 1 for b in t["bounds"]])
                for r in range(k)]
        rows.sort(key=lambda r: len(r[0]["gen"]) - r[2], reverse=True)   # similar remaining lengths per batch
        toks, trunc = decode_batches(model, [t["prompt"] + t["gen"][:cut] for t, _, cut, _ in rows], ctx.gen_bs,
                                     MAX_NEW, ctx.eos, ctx.pad, budgets=[MAX_NEW - cut for _, _, cut, _ in rows],
                                     seed_fn=lambda c, g0=g0: seed_for(run_id, tag, f"{g0}:{c}"),
                                     deadline=deadline, sample=True)
        if len(toks) < len(rows):
            break                                                              # out of time: keep whole traces only
        n_rows, n_tok = n_rows + len(rows), n_tok + sum(map(len, toks))
        succ = {}
        for (t, j, cut, _), tk, tr in zip(rows, toks, trunc):
            text = ctx.tok.decode(t["gen"][:cut] + tk, skip_special_tokens=True)
            succ.setdefault((t["pid"], j), []).append(primary_ok(text, ctx.gold[t["pid"]], tr))
        for t in group:
            done[t["pid"]] = [sum(succ[(t["pid"], j)]) / k for j in range(len(t["bounds"]) + 1)]
    return done, n_rows, n_tok


def mca(c3, d):
    ok, deadline = c3.budget("MCa")
    if not ok:
        return {"MCa_skipped": SKIP}
    ctx = c3.s
    greedy = c3.src_load("S2", "greedy")
    greedy = random.Random(0).sample(greedy, len(greedy))   # a time cut then keeps a mix of main and p1, not main only
    t0 = time.time()
    raw, n_rows, n_tok = mc_success(ctx, greedy, c3.run_id, "mc", deadline)
    done = {p: {"mc": mc, "step_y": [int(mc[j] <= 1 / 3 and mc[j - 1] >= 2 / 3) for j in range(1, len(mc))]}
            for p, mc in raw.items()}
    ctx.save(d, "mc", done)
    by = {t["pid"]: t for t in greedy}
    wrong = [p for p in done if not by[p]["ok"]]
    return {"mc_traces": len(done), "mc_of": len(greedy), "mc_completions": n_rows,
            "mc_tok_per_s": n_tok / max(time.time() - t0, 1e-9),
            "mc_step_pos": sum(sum(v["step_y"]) for v in done.values()),
            "mc_traces_with_bad_step": sum(1 for v in done.values() if any(v["step_y"])),
            "mc_wrong_traces": len(wrong),
            "mc_wrong_with_bad_step": sum(1 for p in wrong if any(done[p]["step_y"])),
            "mc_prompt_success_correct": statistics.mean([done[p]["mc"][0] for p in done if by[p]["ok"]] or [float("nan")]),
            "mc_prompt_success_wrong": statistics.mean([done[p]["mc"][0] for p in wrong] or [float("nan")])}


def mcprobe(c3, d):
    ok, deadline = c3.budget("MCprobe")
    mc_path = c3.run_dir / "stage_MCa" / "mc.pt"
    if not ok or not mc_path.exists():
        return {"MCprobe_skipped": SKIP if not ok else "skipped: no MC labels"}
    ctx = c3.s
    mc = torch.load(mc_path, weights_only=False)
    traces = [t for t in c3.src_load("S2", "greedy") if t["pid"] in mc]
    L = ctx.student().config.num_hidden_layers
    layers = [L // 2, 3 * L // 4, L]
    feats = ctx.feats(traces, layers)
    ctx.save(d, "feats", {"pids": [t["pid"] for t in traces], "layers": layers, "feats": [f.half() for f in feats]})
    pr = P.fit_step(feats, [mc[t["pid"]]["step_y"] for t in traces], [not t["ok"] for t in traces],
                    [c3.group[t["pid"]] for t in traces], layers,
                    layer_sets={str(layers[1]): [layers[1]], str(layers[2]): [layers[2]],
                                f"{layers[1]}+{layers[2]}": layers[1:]}, deadline=deadline)
    m = dict(pr["metrics"])
    if pr["name"] is not None and time.time() < deadline:
        pr["layers_all"] = layers
        m["step_eval_fpr"] = _eval_fpr(ctx, pr, c3.src_load("S1", "a0"), lambda f: P.flag(pr, f))
    ctx.save(d, "probe", pr)
    ctx.drop_fp32()
    return m


def b1(c3, d):
    ok, _ = c3.budget("B1")
    if not ok:
        return {"B1_skipped": SKIP}
    ctx = c3.b
    tea = {t["pid"]: t for t in c3.src_load("S1", "teacher")}   # load the reused teacher numbers before generating
    tea_tok = load_tokenizer(ctx.cfg["teacher"])
    same = sum(prompt_ids(ctx.tok, p["question"]) == prompt_ids(tea_tok, p["question"]) for p in ctx.test[:20])
    model = ctx.student()
    stu, rate = ctx.greedy(model, ctx.test, MAX_NEW)
    ctx.save(d, "a0", stu)
    m = {"b1_tok_per_s": rate, "b1_n": len(stu), "b1_template_identical": same}
    for s in ("p2", "main", "p1", "rc"):
        sub = [t for t in stu if t["set"] == s]
        m[f"b1_student_{s}_primary"] = _rate(sub, "ok")
        m[f"b1_student_{s}_strict"] = _rate(sub, "ok_strict")
        m[f"b1_teacher_{s}_primary"] = _rate([tea[t["pid"]] for t in sub], "ok")
        m[f"b1_student_{s}_trunc"] = _rate(sub, "truncated")
    diff = [int(tea[t["pid"]]["ok"]) - int(t["ok"]) for t in stu if t["set"] == "p2"]
    m["b1_gap_p2"] = statistics.mean(diff)
    m["b1_gap_p2_se"] = statistics.stdev(diff) / len(diff) ** 0.5
    return m


def _rollouts(c3, ctx, probs, name, deadline):
    """greedy + 1 sample per problem, chunk by chunk (both runs of a chunk before the next chunk starts), until the
    deadline; probs are shuffled, so a time cut keeps every set represented."""
    model, bs = ctx.student(), ctx.gen_bs
    t0 = time.time()
    greedy, sampled = [], []
    for i in range(0, len(probs), bs):
        if time.time() > deadline:
            break
        chunk = probs[i:i + bs]
        prompts = [prompt_ids(ctx.tok, p["question"]) for p in chunk]
        g, gt, _ = decode(model, prompts, MAX_NEW, ctx.eos, ctx.pad)
        s, st, _ = decode(model, prompts, MAX_NEW, ctx.eos, ctx.pad, sample=True,
                          seed=seed_for(c3.run_id, name, chunk[0]["pid"]))
        greedy += [ctx.trace(p["pid"], pr, x, t) for p, pr, x, t in zip(chunk, prompts, g, gt)]
        sampled += [ctx.trace(p["pid"], pr, x, t) for p, pr, x, t in zip(chunk, prompts, s, st)]
    sec = max(time.time() - t0, 1e-9)
    both = greedy + sampled
    return greedy, sampled, {"n_problems": len(greedy), "of": len(probs),
                             "tok_per_s": sum(len(t["gen"]) for t in both) / sec, "sec": sec,
                             **{f"greedy_primary_{s}": _rate(greedy, "ok", s) for s in ("main", "p1", "p2", "rc")},
                             "greedy_primary": _rate(greedy, "ok"), "sampled_primary": _rate(sampled, "ok"),
                             "trunc": _rate(both, "truncated"),
                             "mean_tokens": statistics.mean(len(t["gen"]) for t in both) if both else float("nan")}


def b2(c3, d):
    ok, deadline = c3.budget("B2")
    if not ok:
        return {"B2_skipped": SKIP}
    greedy, sampled, m = _rollouts(c3, c3.b, c3.train_b, "b2", deadline)
    c3.b.save(d, "traces", greedy + sampled)
    return {f"b2_{k}": v for k, v in m.items()}


def _probe(c3, ctx, traces, layers, layer_sets, eval_traces, holdout=None, c_grid=P.C_GRID, feats=None,
           deadline=None):
    feats = feats if feats is not None else ctx.feats(traces, layers)
    pr = P.fit(feats, [not t["ok"] for t in traces], [c3.group[t["pid"]] for t in traces], layers,
               layer_sets=layer_sets, c_grid=c_grid, trace_sets=_sets(traces), eval_share=EVAL_SHARE,
               holdout=holdout, deadline=deadline)
    if pr["name"] is not None and eval_traces is not None and (deadline is None or time.time() < deadline):
        pr["layers_all"] = layers
        pr["metrics"]["probe_eval_fpr"] = _eval_fpr(ctx, pr, eval_traces, lambda f: P.flag(pr, f))
    return pr, feats


def b3(c3, d):
    ok, deadline = c3.budget("B3")
    path = c3.run_dir / "stage_B2" / "traces.pt"
    if not ok or not path.exists():
        return {"B3_skipped": SKIP if not ok else "skipped: no 4B rollouts"}
    ctx = c3.b
    traces = torch.load(path, weights_only=False)
    L = ctx.student().config.num_hidden_layers
    layers = [3 * L // 4, L]
    a0 = torch.load(c3.run_dir / "stage_B1" / "a0.pt", weights_only=False) \
        if (c3.run_dir / "stage_B1" / "a0.pt").exists() else None
    try:
        pr, _ = _probe(c3, ctx, traces, layers, None, a0, c_grid=B3_C_GRID, deadline=deadline)
    finally:
        ctx.drop_fp32()
        ctx._student = None   # free the 4B before part (c), also if the probe failed
        smoke.free()
    ctx.save(d, "probe", pr)
    return {f"b3_{k}": v for k, v in pr["metrics"].items()}


def c2(c3, d):
    ok, deadline = c3.budget("C2")
    if not ok:
        return {"C2_skipped": SKIP}
    greedy, sampled, m = _rollouts(c3, c3.s, c3.train_c, "c2", deadline)
    c3.s.save(d, "traces", greedy + sampled)
    return {f"c2_{k}": v for k, v in m.items()}


def c3_(c3, d):
    ok, deadline = c3.budget("C3")
    path = c3.run_dir / "stage_C2" / "traces.pt"
    if not ok or not path.exists():
        return {"C3_skipped": SKIP if not ok else "skipped: no 1.7B scaling rollouts"}
    ctx = c3.s
    traces = torch.load(path, weights_only=False)
    L = ctx.student().config.num_hidden_layers
    layers = [L // 2, 3 * L // 4, L]
    feats = ctx.feats(traces, layers)
    ctx.save(d, "feats", {"pids": [t["pid"] for t in traces], "sets": _sets(traces), "layers": layers,
                          "wrong": [not t["ok"] for t in traces], "groups": [c3.group[t["pid"]] for t in traces],
                          "feats": [f.half() for f in feats]})
    groups = [c3.group[t["pid"]] for t in traces]
    hold = holdout_by_oid(groups)                       # fixed held-out groups for every curve point
    train_groups = sorted({g for g in groups if g not in hold})
    random.Random(0).shuffle(train_groups)
    wrong = [not t["ok"] for t in traces]
    curve, m = [], {}
    for total in (300, 600, 1200, 2400):
        if time.time() > deadline:
            curve.append({"total": total, "skipped": "stage time budget"})
            continue
        n_train = int(total * 0.8)
        keep, n = set(), 0
        for g in train_groups:
            if n >= n_train:
                break
            keep.add(g)
            n += groups.count(g)
        idx = [i for i, g in enumerate(groups) if g in keep or g in hold]
        sub = lambda xs: [xs[i] for i in idx]
        pr = P.fit(sub(feats), sub(wrong), sub(groups), layers, layer_sets={"21": [layers[1]]},
                   trace_sets=sub(_sets(traces)), eval_share=EVAL_SHARE, holdout=hold, deadline=deadline)
        pm = pr["metrics"]
        curve.append({"total": total, "train_traces": pm.get("probe_train_traces"),
                      "heldout_traces": pm.get("probe_holdout_traces"), "C": pm.get("probe_C"),
                      "auroc": pm.get("probe_auroc"), "recall": pm.get("probe_recall"),
                      "by_set": {s: v["auroc"] for s, v in pm.get("probe_by_set", {}).items()}})
    m["c3_curve"] = curve
    if time.time() + 330 < deadline:                    # full protocol at full size (~5.5 min), plus eval FPR per set
        pr, _ = _probe(c3, ctx, traces, layers, {"21": [layers[1]], "28": [layers[2]], "21+28": layers[1:]},
                       c3.src_load("S1", "a0"), holdout=hold, feats=feats, deadline=deadline)
        ctx.save(d, "probe", pr)
        m.update({f"c3_{k}": v for k, v in pr["metrics"].items()})
    ctx.drop_fp32()
    return m


STAGES = [("MCa", mca), ("MCprobe", mcprobe), ("B1", b1), ("B2", b2), ("B3", b3), ("C2", c2), ("C3", c3_)]
