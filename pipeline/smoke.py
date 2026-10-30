"""SMOKE stages S1-S8 (§10). Pipeline check only: sizes are fixed by the spec, accuracy numbers carry no weight.

Artifacts per stage go to runs/<RUN_ID>/stage_<name>/ as token ids, pids, flags, probe and adapter (never question
text). A restarted pod reloads them, so every stage can run after a restart once its inputs are on the volume.
"""
import gc
import json
import math
import shutil
import statistics
import time

import torch

import data as D
import probe as P
from continuations import continuations
from core import arm_metrics, boundaries, extract_lenient, extract_strict, kill_plan, matches, rollback_cut, seed_for
from eval import plain, retry
from generate import decode, decode_batches, eos_ids, load_model, load_tokenizer, prefill_states, prompt_ids
from teacher_score import score
from train_repair import lora_state, set_lora, train, wrap_lora

SETS = ("p2", "main", "p1", "rc")
TEST_SIZES = {"p2": 200, "main": 40, "p1": 40, "rc": 100}
TRAIN_N, IDENTITY_N, CONT_K, CONT_MAX = 300, 20, 4, 150
TRUNC_LIMIT, MAX_NEW_RETRY = 0.10, 800


def free():
    gc.collect()
    torch.cuda.empty_cache()


class Ctx:
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        from huggingface_hub import snapshot_download
        self.cfg, self.run_id, self.run_dir = cfg, run_id, run_dir
        self.gen_bs, self.pre_bs = tiers[tier]["gen_batch"], tiers[tier]["teacher_prefill_batch"]
        gsm = D.load_gsm(snapshot_download(cfg["dataset"], repo_type="dataset", local_files_only=True))
        train_oids, test_oids = D.make_split(gsm)
        (run_dir / "data").mkdir(exist_ok=True)
        (run_dir / "data" / "split.json").write_text(json.dumps(
            {"seed": 0, "train_oids": sorted(train_oids), "test_oids": sorted(test_oids)}))
        test = [p for c in ("p2", "main", "p1") for p in D.sample(D.problems(gsm, [c], test_oids), TEST_SIZES[c])]
        self.test = sorted(test + D.random_calculation(TEST_SIZES["rc"], seed=0), key=lambda p: p["pid"])
        self.train = D.sample(D.problems(gsm, ["main", "p1"], train_oids), TRAIN_N)
        assert not {p["oid"] for p in self.train} & {p["oid"] for p in self.test if p["oid"] is not None}
        self.gold = {p["pid"]: p["gold"] for p in self.test + self.train}
        self.tok = load_tokenizer(cfg["student"])
        self.pad = self.tok.pad_token_id
        self._student, self._lora = None, False

    # models
    def student(self):
        if self._student is None:
            self._student = load_model(self.cfg["student"])
            self.eos = eos_ids(self._student)
        return self._student

    def lora(self):
        """The student wrapped with the LoRA adapter (once); its untrained state is kept in stage_S4/lora_init.pt."""
        if not self._lora:
            self._student = wrap_lora(self.student(), seed_for(self.run_id, "lora", 0))
            init = self.run_dir / "stage_S4" / "lora_init.pt"
            init.parent.mkdir(exist_ok=True)
            if init.exists():
                set_lora(self._student, torch.load(init, weights_only=False))
            else:
                torch.save(lora_state(self._student), init)
            self._lora = True
        return self._student

    def lora_init(self):
        return torch.load(self.run_dir / "stage_S4" / "lora_init.pt", weights_only=False)

    # traces
    def max_new(self):
        return json.loads((self.run_dir / "stage_S1" / "metrics.json").read_text())["s1_max_new_final"]

    def trace(self, pid, prompt, gen, truncated):
        text = self.tok.decode(gen, skip_special_tokens=True)
        g = self.gold[pid]
        return {"pid": pid, "set": pid.split(":")[0], "prompt": prompt, "gen": gen, "truncated": truncated,
                "bounds": boundaries([self.tok.decode([t]) for t in gen]),
                "ok": matches(extract_strict(text), g), "ok_lenient": matches(extract_lenient(text), g)}

    def greedy(self, model, probs, max_new, adapter_on=False):
        prompts = [prompt_ids(self.tok, p["question"]) for p in probs]
        toks, trunc = decode_batches(model, prompts, self.gen_bs, max_new, eos_ids(model), self.pad,
                                     adapter_on=adapter_on)
        return [self.trace(p["pid"], pr, g, t) for p, pr, g, t in zip(probs, prompts, toks, trunc)]

    def save(self, d, name, obj):
        torch.save(obj, d / f"{name}.pt")

    def load(self, stage, name):
        return torch.load(self.run_dir / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def layers(self):
        L = self.student().config.num_hidden_layers
        return [L // 2, 3 * L // 4, L]

    def feats(self, model, traces):
        return prefill_states(model, [t["prompt"] + t["gen"] for t in traces],
                              [[len(t["prompt"]) + b for b in t["bounds"]] for t in traces], self.layers(), self.pre_bs)


def _rate(traces, key, s=None):
    xs = [t[key] for t in traces if s is None or t["set"] == s]
    return sum(xs) / len(xs) if xs else float("nan")


def s1(ctx, d):
    """Headroom: A0 student and teacher greedy on the test problems; truncation rule 640 -> 800."""
    m, tried = {}, []
    for max_new in dict.fromkeys([ctx.cfg["max_new_tokens"], MAX_NEW_RETRY]):
        t0 = time.time()
        stu = ctx.greedy(ctx.student(), ctx.test, max_new)
        t1 = time.time()
        teacher = load_model(ctx.cfg["teacher"])
        tea = ctx.greedy(teacher, ctx.test, max_new)
        del teacher
        free()
        m[f"s1_student_sec_{max_new}"], m[f"s1_teacher_sec_{max_new}"] = t1 - t0, time.time() - t1
        trunc = {f"{who}_{s}": _rate(tr, "truncated", s) for who, tr in (("student", stu), ("teacher", tea))
                 for s in SETS}
        m[f"s1_trunc_{max_new}"] = trunc
        tried.append(max_new)
        if max(trunc.values()) <= TRUNC_LIMIT:
            break
    m["s1_max_new_tried"], m["s1_max_new_final"] = tried, tried[-1]
    for who, tr in (("student", stu), ("teacher", tea)):
        for s in SETS:
            m[f"s1_{who}_{s}_strict"] = _rate(tr, "ok", s)
            m[f"s1_{who}_{s}_lenient"] = _rate(tr, "ok_lenient", s)
    diff = [int(t["ok"]) - int(u["ok"]) for t, u in zip(tea, stu) if u["set"] == "p2"]
    m["s1_gap_p2"] = statistics.mean(diff)
    m["s1_gap_p2_se"] = statistics.stdev(diff) / math.sqrt(len(diff))
    m["s1_n"] = {s: sum(t["set"] == s for t in stu) for s in SETS}
    ctx.save(d, "a0", stu)
    ctx.save(d, "teacher", tea)
    return m


def s2(ctx, d):
    """Student rollouts on train prompts: greedy + 1 sample."""
    model, max_new = ctx.student(), ctx.max_new()
    t0 = time.time()
    greedy = ctx.greedy(model, ctx.train, max_new)
    prompts = [t["prompt"] for t in greedy]
    toks, trunc = decode_batches(model, prompts, ctx.gen_bs, max_new, ctx.eos, ctx.pad, sample=True,
                                 seed_fn=lambda c: seed_for(ctx.run_id, "s2", greedy[c * ctx.gen_bs]["pid"]))
    sampled = [ctx.trace(t["pid"], t["prompt"], g, tr) for t, g, tr in zip(greedy, toks, trunc)]
    ctx.save(d, "greedy", greedy)
    ctx.save(d, "sampled", sampled)
    both = greedy + sampled
    return {"s2_n": len(both), "s2_greedy_strict": _rate(greedy, "ok"), "s2_sampled_strict": _rate(sampled, "ok"),
            "s2_mean_tokens": statistics.mean(len(t["gen"]) for t in both),
            "s2_mean_boundaries": statistics.mean(len(t["bounds"]) for t in both),
            "s2_trunc": _rate(both, "truncated"), "s2_sec": time.time() - t0}


def _state_tolerance(ctx, model, traces):
    """Max relative L2 gap between generation-time and prefill hidden states at boundaries (same tokens)."""
    layers = ctx.layers()
    toks, _, states = decode(model, [t["prompt"] for t in traces], ctx.max_new(), ctx.eos, ctx.pad, capture=layers)
    seqs, pos, kept = [], [], []
    for t, g in zip(traces, toks):
        b = [x for x in boundaries([ctx.tok.decode([y]) for y in g]) if x + 1 < len(states)]
        seqs.append(t["prompt"] + g)
        pos.append([len(t["prompt"]) + x for x in b])
        kept.append(b)
    pre = prefill_states(model, seqs, pos, layers, ctx.pre_bs)
    worst = 0.0
    for r, b in enumerate(kept):
        for j, x in enumerate(b):
            for li in range(len(layers)):
                g, p = states[x + 1][li][r], pre[r][j, li]
                worst = max(worst, float((g - p).norm() / p.norm()))
    return worst


def s3(ctx, d):
    """Probe fit/select/tau on S2 traces (labels: strict answer wrong), restart share, state tolerance."""
    model = ctx.student()
    greedy, sampled = ctx.load("S2", "greedy"), ctx.load("S2", "sampled")
    traces = greedy + sampled
    oid = {p["pid"]: p["oid"] for p in ctx.train}
    feats = ctx.feats(model, traces)
    pr = P.fit(feats, [not t["ok"] for t in traces], [oid[t["pid"]] for t in traces], ctx.layers())
    flags = {t["pid"]: P.flag(pr, f) for t, f in zip(greedy, feats[:len(greedy)])}
    ctx.save(d, "probe", pr)
    ctx.save(d, "flags", flags)
    m = dict(pr["metrics"])
    m["s3_state_tol"] = _state_tolerance(ctx, model, greedy[:5])
    m["s3_flagged_greedy"] = sum(1 for v in flags.values() if v)
    m["s3_flagged_wrong_greedy"] = sum(1 for t in greedy if flags[t["pid"]] and not t["ok"])
    return m


def _retry_items(greedy, flags, n):
    return [t for t in greedy if flags.get(t["pid"])][:n]


def s4(ctx, d):
    """Identity: untrained A3 == A2 on flagged traces, identical batches and seeds."""
    model = ctx.lora()
    flags = ctx.load("S3", "flags")
    a0 = _retry_items(ctx.load("S2", "greedy"), flags, IDENTITY_N)
    kw = dict(bs=ctx.gen_bs, max_new=ctx.max_new(), eos=ctx.eos, pad=ctx.pad, run_id=ctx.run_id)
    a2 = retry(model, a0, flags, adapter_on=False, **kw)
    a3 = retry(model, a0, flags, adapter_on=True, **kw)
    diffs = [sum(x != y for x, y in zip(a2[p][0], a3[p][0])) + abs(len(a2[p][0]) - len(a3[p][0])) for p in a2]
    return {"identity_n": len(a2), "identity_equal": all(x == 0 for x in diffs),
            "identity_max_token_diff": max(diffs, default=0)}


def s5(ctx, d):
    """Continuations x4 from flagged prefixes of wrong greedy train traces (<= 150 traces)."""
    model, max_new = ctx.lora(), ctx.max_new()
    flags, greedy = ctx.load("S3", "flags"), ctx.load("S2", "greedy")
    wrong = [t for t in greedy if flags.get(t["pid"]) and not t["ok"]]
    items = []
    for t in wrong[:CONT_MAX // CONT_K]:
        cut = rollback_cut(t["bounds"], flags[t["pid"]])
        items.append({"pid": t["pid"], "prefix": t["prompt"] + t["gen"][:cut], "cut": cut, "kept": t["gen"][:cut]})
    conts = continuations(model, items, CONT_K, ctx.gen_bs, max_new, ctx.eos, ctx.pad, ctx.run_id)
    kept = {it["pid"]: it["kept"] for it in items}
    for c in conts:
        c["ok"] = ctx.trace(c["pid"], [], kept[c["pid"]] + c["cont"], c["truncated"])["ok"]
    ctx.save(d, "conts", conts)
    return {"s5_flagged_wrong": len(wrong), "s5_problems": len(items), "s5_continuations": len(conts),
            "s5_strict": _rate(conts, "ok"), "s5_trunc": _rate(conts, "truncated")}


def s6(ctx, d):
    """Teacher (and sampling-time student) log-probs per continuation token; teacher-also-wrong on those problems."""
    conts = ctx.load("S5", "conts")
    lp_s = score(ctx.lora(), conts, ctx.pre_bs, adapter_on=False)   # untrained adapter == base model
    teacher = load_model(ctx.cfg["teacher"])
    t0 = time.time()
    lp_t = score(teacher, conts, ctx.pre_bs)
    prefill_sec, prefill_tokens = time.time() - t0, sum(len(c["prefix"]) + len(c["cont"]) for c in conts)
    pids = sorted({c["pid"] for c in conts})
    by = {p["pid"]: p for p in ctx.train}
    tea = ctx.greedy(teacher, [by[p] for p in pids], ctx.max_new()) if pids else []
    del teacher
    free()
    ctx.save(d, "logprobs", {"lp_s": lp_s, "lp_t": lp_t})
    adv = [float((-(s - t)).clamp(-5, 5).mean()) for s, t in zip(lp_s, lp_t) if len(s)]
    return {"s6_n": len(conts), "s6_teacher_also_wrong": _rate([{"w": not t["ok"]} for t in tea], "w"),
            "s6_mean_adv": statistics.mean(adv) if adv else float("nan"),
            "s6_prefill_tok_per_s": prefill_tokens / prefill_sec if conts else float("nan")}


def s7(ctx, d):
    """Train one rkl round; kill at kill_at and resume from the last checkpoint (spec 15 / every 10)."""
    model = ctx.lora()
    conts, lp = ctx.load("S5", "conts"), ctx.load("S6", "logprobs")
    data = [{"prefix": c["prefix"], "cont": c["cont"], "lp_t": lp["lp_t"][i], "lp_s": lp["lp_s"][i]}
            for i, c in enumerate(conts) if c["cont"]]
    m = {"train_examples": len(data), "adapter_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if not data:
        torch.save(ctx.lora_init(), d / "adapter.pt")
        return {**m, "train_steps": 0, "train_blocker": "no repair data (no flagged wrong train traces)"}
    ck = d / "ckpt"
    shutil.rmtree(ck, ignore_errors=True)
    set_lora(model, ctx.lora_init())
    steps = math.ceil(len(data) / 16)
    plan = kill_plan(steps)
    t0 = time.time()
    if plan:
        c, k = plan
        first, _ = train(model, data, ck, ctx.run_id, stop_at=k, ckpt_every=c)
        set_lora(model, ctx.lora_init())   # the "killed" process's weights are gone; resume must restore them
        losses, _ = train(model, data, ck, ctx.run_id, resume=True, ckpt_every=c)
        gap = [abs(first[s] - losses[s]) / max(1.0, abs(first[s])) for s in range(c + 1, k + 1)]
        m.update({"train_ckpt_every": c, "train_kill_at": k, "resume_max_rel_diff": max(gap),
                  "resume_ok": max(gap) <= 1e-3})
    else:
        losses, _ = train(model, data, ck, ctx.run_id, ckpt_every=10)
        m.update({"resume_ok": None, "train_blocker": f"only {steps} steps: kill/resume check needs >= 5"})
    sec = time.time() - t0
    torch.save(lora_state(model), d / "adapter.pt")
    return {**m, "train_steps": steps, "loss_start": losses[1], "loss_end": losses[steps],
            "train_steps_per_s": steps / sec, "train_sec": sec}


def s8(ctx, d):
    """Eval A0, A2, A3, A6 on the S1 test problems."""
    model, max_new = ctx.lora(), ctx.max_new()
    set_lora(model, torch.load(ctx.run_dir / "stage_S7" / "adapter.pt", weights_only=False))
    a0, pr = ctx.load("S1", "a0"), ctx.load("S3", "probe")
    t0 = time.time()
    flags = {t["pid"]: P.flag(pr, f) for t, f in zip(a0, ctx.feats(model, a0))}
    kw = dict(bs=ctx.gen_bs, max_new=max_new, eos=ctx.eos, pad=ctx.pad)
    a2 = retry(model, a0, flags, run_id=ctx.run_id, adapter_on=False, **kw)
    a3 = retry(model, a0, flags, run_id=ctx.run_id, adapter_on=True, **kw)
    a6 = plain(model, a0, adapter_on=True, **kw)
    sec = time.time() - t0
    base = {t["pid"]: t for t in a0}
    arms = {"A0": {}, "A2": a2, "A3": a3, "A6": a6}
    out = {}
    for arm, over in arms.items():
        rows = []
        for t in a0:
            gen, trunc = over.get(t["pid"], (t["gen"], t["truncated"]))
            r = ctx.trace(t["pid"], t["prompt"], gen, trunc) if t["pid"] in over else t
            rows.append({"set": t["set"], "a0_ok": base[t["pid"]]["ok"], "ok": r["ok"], "ok_lenient": r["ok_lenient"],
                         "flagged": bool(flags[t["pid"]]) and arm in ("A2", "A3"), "a0_tokens": len(t["gen"]),
                         "tokens": len(gen), "truncated": trunc})
        out[arm] = rows
    ctx.save(d, "arms", out)
    m ={"s8_sec": sec, "s8_traces": sum(len(v) for v in (a2, a3, a6)), "s8_flagged": sum(1 for v in flags.values() if v)}
    for arm, rows in out.items():
        m[f"eval_{arm}"] = arm_metrics(rows)
        for s in SETS:
            sub = [r for r in rows if r["set"] == s]
            m[f"eval_{arm}_{s}_strict"] = _rate(sub, "ok")
            m[f"eval_{arm}_{s}_lenient"] = _rate(sub, "ok_lenient")
    return m


STAGES = [("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S6", s6), ("S7", s7), ("S8", s8)]
