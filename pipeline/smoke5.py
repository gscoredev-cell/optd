"""SMOKE-5 (diagnostic, one pod, <= $4, 120-min cap): detector test with Qwen3-8B as the student; no teacher loaded.

D1  4 samples (T 0.8, top_p 0.95, cap 800) per problem. Train: RC3 150 + RC4 150 (fresh seeds 503/504) + p2 100
    (train templates). Test: RC3 100 + RC4 100 (seeds 603/604) + p2 100 (test templates), the p2 drawn from SMOKE-2 S1's
    200 p2 test problems, whose Qwen3-8B greedy trace (seeded into the generation cache) is sample 0: p_hat there mixes
    1 greedy + 3 sampled. p_hat distribution and contested share (1..3 of 4 correct) per set.
D2  target a, within-problem probe: one wrong + one correct trace per contested train problem; bf16 prefill features
    (batch 8) at layers L/2, 3L/4, L (18, 27, 36), saved; layer, C and pooling by grouped CV on train groups; tau at
    FPR 10% on held-out train contested correct traces.
D4a (before the MC stages, so their time cannot cost it) within-pair AUROC (+ CI) on fresh pairs of contested TEST
    problems, by set and boundary position; FPR / recall per set at D2's tau on all their traces.
D3  target b, recoverability: <= 120 contested train traces (60 D2 pairs), 3 completions from the prompt and from each
    boundary; boundary label unrecoverable = MC success < 0.5 (recoverable = >= 0.5); boundary probe by grouped CV,
    held-out AUROC by boundary position; tau at FPR 10% on held-out recoverable boundaries; first point with MC < 0.25
    (SMOKE-4's rule) and restart share for the wrong traces.
D4b <= 60 contested TEST traces (30 pairs) labelled the same way: boundary AUROC of D3's probe by set and position;
    FPR / recall per set at D3's tau.
Every generation goes through the generation cache (gencache.py); hits and misses are reported per stage.
"""
import collections
import pathlib
import random
import statistics
import time

import numpy as np
import torch

import data as D
import probe as P
import smoke
from core import auroc, holdout_by_oid, primary_ok, seed_for
from gencache import GREEDY, GenCache, cached_decode
from generate import prefill_states, prompt_ids
from smoke3 import MAX_NEW, MC_GROUP, MC_K, pipeline_deadline
from smoke4 import K, Ctx4, by_problem, contested, phat_dist

STUDENT = "Qwen/Qwen3-8B"
SETS = ("rc3", "rc4", "p2")
SIZES = {("train", "rc3"): 150, ("train", "rc4"): 150, ("train", "p2"): 100,
         ("test", "rc3"): 100, ("test", "rc4"): 100, ("test", "p2"): 100}
RC_SEEDS = {("train", "rc3"): 503, ("train", "rc4"): 504, ("test", "rc3"): 603, ("test", "rc4"): 604}  # used: 0 103-105 200
D3_PAIRS, D4B_PAIRS, LABEL, DROP, FEAT_BS, POSITIONS = 60, 30, 0.5, 0.25, 8, 12
# minutes: A100 estimates x 1.3 (sum 93 of ~104 usable; D2 fits 3 layers x 4096 dims) and the least time a stage
# may start with
BUDGET_MIN = {"D1": 30, "D2": 10, "D4a": 2, "D3": 34, "D4b": 17}
START_MIN = {"D1": 10, "D2": 4, "D4a": 1, "D3": 8, "D4b": 4}
SKIP = "skipped: not enough pod time left"


def sid(t):
    return f"{t['pid']}#{t['k']}"


class Ctx5:
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        self.cfg, self.run_id, self.run_dir = cfg, run_id, run_dir
        self.s = smoke.Ctx({**cfg, "student": STUDENT}, tiers, tier, run_id, run_dir)
        self.s.gen_bs *= 2      # a multiple of K; 8B KV at cap 800 ~ 18 GB for 128 rows (tier 80), ~ 9 GB for 64 (48)
        self.cache = GenCache(STUDENT)
        gsm = Ctx4._gsm(cfg)
        oids = dict(zip(("train", "test"), D.make_split(gsm)))
        s1_p2 = [p for p in self.s.test if p["set"] == "p2"]          # SMOKE-2 S1's 200 p2 test problems
        probs = []
        for (sp, st), n in SIZES.items():
            if st == "p2":
                ps = D.sample(s1_p2 if sp == "test" else D.problems(gsm, ["p2"], oids["train"]), n, seed=5)
            else:
                k = int(st[2])
                ps = [dict(p, pid=f"{st}:{p['pid'].split(':', 1)[1]}", set=st)
                      for p in D.random_calculation(n, seed=RC_SEEDS[(sp, st)], min_ops=k, max_ops=k)]
            probs += [dict(p, split=sp) for p in ps]
        assert not ({p["oid"] for p in probs if p["split"] == "train" and p["oid"] is not None}
                    & {p["oid"] for p in probs if p["split"] == "test" and p["oid"] is not None})
        self.probs = random.Random(5).sample(probs, len(probs))      # a time cut keeps every set
        self.split = {p["pid"]: p["split"] for p in probs}
        self.group = {p["pid"]: f"gsm:{p['oid']}" if p["oid"] is not None else p["pid"] for p in probs}
        self.greedy0 = {p["pid"] for p in probs if p["split"] == "test" and p["set"] == "p2"}
        self.s.gold.update({p["pid"]: p["gold"] for p in probs})

    def budget(self, name, reserve=()):
        """(start allowed, stage deadline); the deadline leaves the reserved stages their nominal budgets."""
        end = pipeline_deadline() - sum(BUDGET_MIN[r] for r in reserve) * 60
        return time.time() + START_MIN[name] * 60 <= end, min(time.time() + BUDGET_MIN[name] * 60, end)

    def load(self, stage, name):
        return torch.load(self.run_dir / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def has(self, stage, name):
        return (self.run_dir / f"stage_{stage}" / f"{name}.pt").exists()

    def gen(self, seqs, seeds, deadline, budgets=None, sample=True):
        model = self.s.student()
        return cached_decode(self.cache, model, seqs, seeds, self.s.gen_bs, MAX_NEW, self.s.eos, self.s.pad,
                             budgets=budgets, deadline=deadline, sample=sample)

    def mark(self):
        return self.cache.hits, self.cache.misses, self.cache.gen_tokens, time.time()

    def since(self, mark, tag):
        h, mi, tok, t0 = mark
        sec = max(time.time() - t0, 1e-9)
        return {f"{tag}_cache_hits": self.cache.hits - h, f"{tag}_cache_misses": self.cache.misses - mi,
                f"{tag}_tok_per_s": (self.cache.gen_tokens - tok) / sec, f"{tag}_sec": sec}


def seed_cache(c5):
    """Import SMOKE-2 S1's Qwen3-8B greedy test traces (cap 800) as greedy cache entries. Their prompt ids came from the
    1.7B tokenizer; check them against the 8B's for the problems D1 uses."""
    src = pathlib.Path("/workspace/runs") / c5.cfg["smoke3_source_run"] / "stage_S1" / "teacher.pt"
    tea = torch.load(src, weights_only=False)
    new = [t for t in tea if c5.cache.get(t["prompt"], *GREEDY, 0, MAX_NEW) is None]
    c5.cache.put_many([(t["prompt"], *GREEDY, 0, MAX_NEW, t["gen"], t["truncated"]) for t in new])
    q = {p["pid"]: p for p in c5.probs}
    used = [t for t in tea if t["pid"] in q]
    return {"d1_seed_traces": len(tea), "d1_seed_imported": len(new), "d1_seed_used": len(used),
            "d1_seed_prompt_ids_same": sum(prompt_ids(c5.s.tok, q[t["pid"]]["question"]) == t["prompt"] for t in used)}


def d1(c5, d):
    ok, deadline = c5.budget("D1", reserve=("D2", "D4a"))
    if not ok:
        return {"D1_skipped": SKIP}
    ctx = c5.s
    m = seed_cache(c5)
    mk = c5.mark()
    prompts = {p["pid"]: prompt_ids(ctx.tok, p["question"]) for p in c5.probs}
    rows = [(p["pid"], k) for p in c5.probs for k in range(K)]
    greedy = [r for r in rows if r[1] == 0 and r[0] in c5.greedy0]
    sampled = [r for r in rows if not (r[1] == 0 and r[0] in c5.greedy0)]
    out = {}
    for rs, sample in ((greedy, False), (sampled, True)):
        toks, trunc = c5.gen([prompts[p] for p, _ in rs],
                             [seed_for("smoke5", "d1", f"{p}#{k}") if sample else 0 for p, k in rs], deadline,
                             sample=sample)
        out.update({r: (t, tr) for r, t, tr in zip(rs, toks, trunc) if t is not None})
    m.update(c5.since(mk, "d1"))
    traces = [dict(ctx.trace(p, prompts[p], *out[(p, k)]), k=k, split=c5.split[p], greedy=k == 0 and p in c5.greedy0)
              for p in dict.fromkeys(r[0] for r in rows) if all((p, j) in out for j in range(K)) for k in range(K)]
    ctx.save(d, "traces", traces)
    return {**m, "d1_problems": len(traces) // K, "d1_of": len(c5.probs), "d1_bs": ctx.gen_bs,
            "d1_trunc": smoke._rate(traces, "truncated"), "d1_greedy0": sum(t["greedy"] for t in traces),
            "d1_mean_tokens": statistics.mean(len(t["gen"]) for t in traces) if traces else float("nan"),
            "d1_dist": phat_dist(traces, c5.split, SETS)}


def d2(c5, d):
    ok, deadline = c5.budget("D2", reserve=("D4a",))
    if not ok or not c5.has("D1", "traces"):
        return {"D2_skipped": SKIP if not ok else "skipped: no D1 traces"}
    ctx = c5.s
    pair_tr, test_tr = [], []
    for pid, ts in by_problem(c5.load("D1", "traces")).items():
        if not contested(ts):
            continue
        ts = [t for t in ts if t["bounds"]]
        if c5.split[pid] == "test":
            test_tr += ts
            continue
        w, c = [t for t in ts if not t["ok"]], [t for t in ts if t["ok"]]
        if w and c:
            pair_tr += [w[0], c[0]]
    L = ctx.student().config.num_hidden_layers
    layers = [L // 2, 3 * L // 4, L]
    t0 = time.time()
    tr = pair_tr + test_tr
    feats = prefill_states(ctx.student(), [t["prompt"] + t["gen"] for t in tr],
                           [[len(t["prompt"]) + b for b in t["bounds"]] for t in tr], layers, FEAT_BS)
    feat_sec = time.time() - t0
    n = len(pair_tr)
    ctx.save(d, "feats", {"sids": [sid(t) for t in tr], "n_pair": n, "layers": layers,
                          "feats": [f.bfloat16() for f in feats]})   # lossless (bf16 prefill); fp16 can overflow
    ctx.save(d, "pairs", pair_tr)
    ctx.save(d, "test_traces", test_tr)
    pr = P.fit_within([f.numpy() for f in feats[:n]], [not t["ok"] for t in pair_tr], [t["pid"] for t in pair_tr],
                      [c5.group[t["pid"]] for t in pair_tr], layers, layer_sets={str(l): [l] for l in layers},
                      deadline=deadline)
    ctx.save(d, "probe", pr)
    return {"d2_layers_all": layers, "d2_test_traces": len(test_tr), "d2_feat_sec": feat_sec, **pr["metrics"]}


def pair_by_position(raw, pp):
    """{k: {n, auroc}}: within-pair AUROC of the boundary-k scores over pairs whose twins both reach boundary k."""
    out = {}
    for k in range(1, POSITIONS + 1):
        pk = [(a, b) for a, b in pp if len(raw[a]) >= k and len(raw[b]) >= k]
        if pk:
            out[k] = {"n": len(pk), "auroc": P.pair_auroc([raw[a][k - 1] for a, _ in pk], [raw[b][k - 1] for _, b in pk])}
    return out


def boundary_by_position(scores, labels):
    """{k: {n, pos, auroc}}: boundary AUROC at position k (label 1 = unrecoverable) over traces reaching k."""
    out = {}
    for k in range(1, POSITIONS + 1):
        ix = [i for i in scores if len(scores[i]) >= k]
        pos = [float(scores[i][k - 1]) for i in ix if labels[i][k - 1]]
        neg = [float(scores[i][k - 1]) for i in ix if not labels[i][k - 1]]
        if ix:
            out[k] = {"n": len(ix), "pos": len(pos), "auroc": auroc(pos, neg)}
    return out


def d4a(c5, d):
    ok, _ = c5.budget("D4a")
    if not ok or not c5.has("D2", "probe"):
        return {"D4a_skipped": SKIP if not ok else "skipped: no D2 probe"}
    pr, f, test_tr = c5.load("D2", "probe"), c5.load("D2", "feats"), c5.load("D2", "test_traces")
    if pr.get("name") is None:
        return {"D4a_skipped": "skipped: " + pr["metrics"].get("within_blocker", "no probe")}
    raw = [P._score(pr["model"], x.float().numpy(), pr["cols"]) for x in f["feats"][f["n_pair"]:]]
    s = [float(P.POOLS[pr["pool"]](r)) for r in raw]
    wrong, pids = [not t["ok"] for t in test_tr], [t["pid"] for t in test_tr]
    out = {}
    for st in (*SETS, None):
        idx = [i for i, t in enumerate(test_tr) if st is None or t["set"] == st]
        neg = [s[i] for i in idx if not wrong[i]]
        pos = [s[i] for i in idx if wrong[i]]
        pp = P.pairs(idx, wrong, pids)
        sw, sc = [s[a] for a, _ in pp], [s[b] for _, b in pp]
        out[st or "pooled"] = {"problems": len({pids[i] for i in idx}), "correct": len(neg), "wrong": len(pos),
                               "fpr": float(np.mean(np.array(neg) >= pr["tau"])) if neg else float("nan"),
                               "recall": float(np.mean(np.array(pos) >= pr["tau"])) if pos else float("nan"),
                               "pairs": len(pp), "within_auroc": P.pair_auroc(sw, sc), "within_ci": P.pair_ci(sw, sc)}
    return {"d4a_tau": pr["tau"], "d4a_pool": pr["pool"], "d4a_layers": pr["name"], "d4a_test": out,
            "d4a_by_position": pair_by_position(raw, P.pairs(range(len(test_tr)), wrong, pids))}


def mc_success(c5, traces, tag, deadline, k=MC_K):
    """MC success at the prompt and every boundary of each trace: share of k sampled completions (T 0.8, total cap
    MAX_NEW) correct under the primary rule, through the cache. Groups of MC_GROUP traces; a group cut by the deadline
    is dropped (its finished rows stay cached). Returns ({sid: [mc at prompt, boundary 1, ...]}, completions)."""
    ctx = c5.s
    done, n = {}, 0
    for g0 in range(0, len(traces), MC_GROUP):
        if time.time() > deadline:
            break
        group = traces[g0:g0 + MC_GROUP]
        rows = [(t, j, cut, r) for t in group for j, cut in enumerate([0] + [b + 1 for b in t["bounds"]])
                for r in range(k)]
        rows.sort(key=lambda r: len(r[0]["gen"]) - r[2], reverse=True)   # similar remaining lengths per batch
        toks, trunc = c5.gen([t["prompt"] + t["gen"][:cut] for t, _, cut, _ in rows],
                             [seed_for("smoke5", tag, f"{sid(t)}@{j}#{r}") for t, j, _, r in rows], deadline,
                             budgets=[MAX_NEW - cut for _, _, cut, _ in rows])
        if any(x is None for x in toks):
            break
        n += len(rows)
        succ = collections.defaultdict(list)
        for (t, j, cut, _), tk, tr in zip(rows, toks, trunc):
            text = ctx.tok.decode(t["gen"][:cut] + tk, skip_special_tokens=True)
            succ[(sid(t), j)].append(primary_ok(text, ctx.gold[t["pid"]], tr))
        for t in group:
            done[sid(t)] = [sum(succ[(sid(t), j)]) / k for j in range(len(t["bounds"]) + 1)]
    return done, n


def _unrec(mc):
    """Boundary labels from MC success at boundaries 1..n: 1 = unrecoverable (< LABEL)."""
    return np.array([int(v < LABEL) for v in mc[1:]])


def d3(c5, d):
    ok, deadline = c5.budget("D3", reserve=("D4b",))
    if not ok or not c5.has("D2", "pairs"):
        return {"D3_skipped": SKIP if not ok else "skipped: no contested pairs"}
    pairs_, f = c5.load("D2", "pairs"), c5.load("D2", "feats")      # pairs_ = [wrong, correct, wrong, correct, ...]
    pick = random.Random(0).sample(range(len(pairs_) // 2), min(D3_PAIRS, len(pairs_) // 2))
    sel = [j for p in pick for j in (2 * p, 2 * p + 1)]
    mk = c5.mark()
    mc, n_rows = mc_success(c5, [pairs_[j] for j in sel], "d3", deadline)
    m = {**c5.since(mk, "d3"), "d3_traces": len(mc), "d3_of": len(sel), "d3_completions": n_rows}
    sel = [j for j in sel if sid(pairs_[j]) in mc]
    traces = [pairs_[j] for j in sel]
    c5.s.save(d, "oracle", {sid(t): mc[sid(t)] for t in traces})
    feats = [f["feats"][j].float().numpy() for j in sel]
    y = [_unrec(mc[sid(t)]) for t in traces]
    groups = [c5.group[t["pid"]] for t in traces]
    for name, keep in (("correct", True), ("wrong", False)):
        ys = [y[i] for i, t in enumerate(traces) if t["ok"] == keep]
        m[f"d3_unrec_share_{name}"] = float(np.concatenate(ys).mean()) if ys else float("nan")
    first = {sid(t): next((j for j, v in enumerate(mc[sid(t)]) if v < DROP), None) for t in traces if not t["ok"]}
    hist = collections.Counter(first.values())
    m.update({"d3_wrong_traces": len(first),
              "d3_restart_share": sum(j is not None and j <= 1 for j in first.values()) / len(first) if first else float("nan"),
              "d3_never_share": sum(j is None for j in first.values()) / len(first) if first else float("nan"),
              "d3_mc_prompt_mean": statistics.mean(mc[sid(t)][0] for t in traces) if traces else float("nan"),
              "d3_first_drop_hist": {**{str(j): hist[j] for j in sorted(k for k in hist if k is not None)},
                                     "never": hist[None]}})
    if not traces:
        return m
    pr = P.fit_step(feats, y, [not t["ok"] for t in traces], groups, f["layers"],
                    layer_sets={str(l): [l] for l in f["layers"]}, deadline=deadline)
    c5.s.save(d, "probe", pr)
    m.update({f"d3_{k}": v for k, v in pr["metrics"].items()})
    if pr["name"] is not None:
        hold = holdout_by_oid(groups)                  # the same held-out groups fit_step used
        te = [i for i, g in enumerate(groups) if g in hold]
        m["d3_by_position"] = boundary_by_position({i: P._score(pr["model"], feats[i], pr["cols"]) for i in te},
                                                   {i: y[i] for i in te})
    return m


def d4b(c5, d):
    ok, deadline = c5.budget("D4b")
    if not ok or not c5.has("D3", "probe"):
        return {"D4b_skipped": SKIP if not ok else "skipped: no D3 probe"}
    pr = c5.load("D3", "probe")
    if pr.get("name") is None:
        return {"D4b_skipped": "skipped: " + pr["metrics"].get("step_blocker", "no probe")}
    test_tr, f = c5.load("D2", "test_traces"), c5.load("D2", "feats")
    pp = P.pairs(range(len(test_tr)), [not t["ok"] for t in test_tr], [t["pid"] for t in test_tr])
    sel = [i for a, b in random.Random(0).sample(pp, min(D4B_PAIRS, len(pp))) for i in (a, b)]
    mk = c5.mark()
    mc, n_rows = mc_success(c5, [test_tr[i] for i in sel], "d4b", deadline)
    m = {**c5.since(mk, "d4b"), "d4b_traces": len(mc), "d4b_of": len(sel), "d4b_completions": n_rows}
    sel = [i for i in sel if sid(test_tr[i]) in mc]
    c5.s.save(d, "oracle", {sid(test_tr[i]): mc[sid(test_tr[i])] for i in sel})
    sc = {i: P._score(pr["model"], f["feats"][f["n_pair"] + i].float().numpy(), pr["cols"]) for i in sel}
    y = {i: _unrec(mc[sid(test_tr[i])]) for i in sel}
    out = {}
    for st in (*SETS, None):
        idx = [i for i in sel if st is None or test_tr[i]["set"] == st]
        if not idx:
            continue
        v, lab = np.concatenate([sc[i] for i in idx]), np.concatenate([y[i] for i in idx])
        pos, neg = v[lab == 1], v[lab == 0]
        out[st or "pooled"] = {"traces": len(idx), "boundaries": len(v), "unrec": len(pos),
                               "auroc": auroc(pos.tolist(), neg.tolist()),
                               "fpr": float(np.mean(neg >= pr["tau"])) if len(neg) else float("nan"),
                               "recall": float(np.mean(pos >= pr["tau"])) if len(pos) else float("nan")}
    m.update({"d4b_tau": pr["tau"], "d4b_layers": pr["name"], "d4b_test": out,
              "d4b_by_position": boundary_by_position(sc, y)})
    return m


STAGES = [("D1", d1), ("D2", d2), ("D4a", d4a), ("D3", d3), ("D4b", d4b)]
