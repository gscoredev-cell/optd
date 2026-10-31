"""SMOKE-4 Branch A (diagnostic, one pod, <= 90 min): does Qwen3-1.7B's hidden state separate its own slips from its
successes on the same problem? Branch chosen locally by STAGE 0 on SMOKE-2 features (config smoke4_branch/stage0).

A1 MC stratification: 400 train problems (p1 200, p2 100, main 100; train templates) + 200 p1 test + 200 p2 test,
   4 samples each (T 0.8, cap 800), one shuffled list so a time cut keeps every set. p_hat distribution per set;
   contested = 1..3 of 4 correct (the widest bounds that still give a correct and a wrong trace).
A2 within-problem probe: one wrong + one correct trace per contested train problem; fp32 boundary features, layers
   L/2 and 3L/4 (14, 21), saved; layer, C and pooling by grouped CV on train ids; within-pair AUROC (+ CI) on
   held-out contested problems by pooling and boundary position; tau at FPR 10% on held-out contested correct traces.
A4 (before A3, so the deadline cannot cost it) FPR / recall per set at tau on all traces of contested TEST problems,
   plus within-pair AUROC there.
A3 step localization: <= 100 contested wrong train traces, MC success at the prompt and every boundary
   (3 completions); first point below 0.25; restart share (first drop at the prompt or boundary 1, i.e. rollback to
   the prompt) and histogram; saved as oracle labels.
"""
import collections
import random
import statistics
import time

import numpy as np
import torch

import data as D
import probe as P
import smoke
from core import seed_for
from generate import decode_batches, prompt_ids
from smoke3 import MAX_NEW, mc_success, pipeline_deadline

K = 4                                    # samples per problem in A1
SIZES = {("train", "p1"): 200, ("train", "p2"): 100, ("train", "main"): 100, ("test", "p1"): 200, ("test", "p2"): 200}
A3_MAX, DROP = 100, 0.25
# minutes: nominal budget and the least time a stage may start with (A1 also leaves A2 + A4 their budgets);
# sum 56 x 1.3 = 73 <= ~76 usable
BUDGET_MIN = {"A1": 30, "A2": 8, "A4": 2, "A3": 16}
START_MIN = {"A1": 10, "A2": 4, "A4": 1, "A3": 4}
SKIP = "skipped: not enough pod time left"


class Ctx4:
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        self.run_id, self.run_dir = run_id, run_dir
        self.s = smoke.Ctx(cfg, tiers, tier, run_id, run_dir)    # Qwen3-1.7B
        self.s.gen_bs *= 4                                       # a multiple of K: a problem's samples share a batch
        gsm = self._gsm(cfg)
        oids = dict(zip(("train", "test"), D.make_split(gsm)))
        probs = [dict(p, split=sp) for (sp, c), n in SIZES.items() for p in D.sample(D.problems(gsm, [c], oids[sp]), n)]
        self.probs = random.Random(4).sample(probs, len(probs))
        self.split = {p["pid"]: p["split"] for p in probs}
        self.group = {p["pid"]: f"gsm:{p['oid']}" for p in probs}
        self.s.gold.update({p["pid"]: p["gold"] for p in probs})

    @staticmethod
    def _gsm(cfg):
        from huggingface_hub import snapshot_download
        return D.load_gsm(snapshot_download(cfg["dataset"], repo_type="dataset", local_files_only=True))

    def budget(self, name, reserve=()):
        """(start allowed, stage deadline); the deadline leaves the reserved stages their nominal budgets."""
        end = pipeline_deadline() - sum(BUDGET_MIN[r] for r in reserve) * 60
        return time.time() + START_MIN[name] * 60 <= end, min(time.time() + BUDGET_MIN[name] * 60, end)

    def load(self, stage, name):
        return torch.load(self.run_dir / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def has(self, stage, name):
        return (self.run_dir / f"stage_{stage}" / f"{name}.pt").exists()


def by_problem(traces):
    out = collections.defaultdict(list)
    for t in traces:
        out[t["pid"]].append(t)
    return out


def contested(ts):
    return 0 < sum(t["ok"] for t in ts) < len(ts)


def phat_dist(traces, split):
    """{split_set: {problems, hist of #correct of K, accuracy, contested share}}, plus each split's sets pooled."""
    bp = by_problem(traces)
    out = {}
    for sp in ("train", "test"):
        for s in ("p1", "p2", "main", None):
            ps = [ts for p, ts in bp.items() if split[p] == sp and (s is None or ts[0]["set"] == s)]
            if not ps:
                continue
            hist = collections.Counter(sum(t["ok"] for t in ts) for ts in ps)
            out[f"{sp}_{s or 'all'}"] = {"problems": len(ps), "hist": [hist.get(k, 0) for k in range(K + 1)],
                                         "acc": statistics.mean(t["ok"] for ts in ps for t in ts),
                                         "contested": sum(map(contested, ps)) / len(ps)}
    return out


def a1(c4, d):
    ok, deadline = c4.budget("A1", reserve=("A2", "A4"))
    if not ok:
        return {"A1_skipped": SKIP}
    ctx = c4.s
    model = ctx.student()
    prompts = {p["pid"]: prompt_ids(ctx.tok, p["question"]) for p in c4.probs}
    rows = [p["pid"] for p in c4.probs for _ in range(K)]
    t0 = time.time()
    toks, trunc = decode_batches(model, [prompts[r] for r in rows], ctx.gen_bs, MAX_NEW, ctx.eos, ctx.pad,
                                 seed_fn=lambda c: seed_for(c4.run_id, "a1", c), deadline=deadline, sample=True)
    sec = max(time.time() - t0, 1e-9)
    traces = [dict(ctx.trace(rows[j], prompts[rows[j]], toks[j], trunc[j]), k=j % K, split=c4.split[rows[j]])
              for j in range(len(toks) // K * K)]
    ctx.save(d, "traces", traces)
    return {"a1_problems": len(traces) // K, "a1_of": len(c4.probs), "a1_tok_per_s": sum(map(len, toks)) / sec,
            "a1_sec": sec, "a1_bs": ctx.gen_bs, "a1_trunc": smoke._rate(traces, "truncated"),
            "a1_mean_tokens": statistics.mean(len(t["gen"]) for t in traces) if traces else float("nan"),
            "a1_dist": phat_dist(traces, c4.split)}


def a2(c4, d):
    ok, deadline = c4.budget("A2", reserve=("A4",))
    if not ok or not c4.has("A1", "traces"):
        return {"A2_skipped": SKIP if not ok else "skipped: no A1 traces"}
    ctx = c4.s
    pair_tr, test_tr = [], []
    for pid, ts in by_problem(c4.load("A1", "traces")).items():
        if not contested(ts):
            continue
        ts = [t for t in ts if t["bounds"]]
        if c4.split[pid] == "test":
            test_tr += ts
            continue
        w, c = [t for t in ts if not t["ok"]], [t for t in ts if t["ok"]]
        if w and c:
            pair_tr += [w[0], c[0]]
    L = ctx.student().config.num_hidden_layers
    layers = [L // 2, 3 * L // 4]
    feats = ctx.feats(pair_tr + test_tr, layers)
    ctx.drop_fp32()
    n = len(pair_tr)
    ctx.save(d, "feats", {"sids": [f"{t['pid']}#{t['k']}" for t in pair_tr + test_tr], "n_pair": n,
                          "layers": layers, "feats": [f.half() for f in feats]})
    ctx.save(d, "pairs", pair_tr)
    ctx.save(d, "test_traces", test_tr)
    pr = P.fit_within(feats[:n], [not t["ok"] for t in pair_tr], [t["pid"] for t in pair_tr],
                      [c4.group[t["pid"]] for t in pair_tr], layers,
                      layer_sets={str(l): [l] for l in layers}, deadline=deadline)
    ctx.save(d, "probe", pr)
    return {"a2_layers_all": layers, "a2_test_traces": len(test_tr), **pr["metrics"]}


def a4(c4, d):
    ok, _ = c4.budget("A4")
    if not ok or not c4.has("A2", "probe"):
        return {"A4_skipped": SKIP if not ok else "skipped: no A2 probe"}
    pr, f, test_tr = c4.load("A2", "probe"), c4.load("A2", "feats"), c4.load("A2", "test_traces")
    if pr.get("name") is None:
        return {"A4_skipped": "skipped: " + pr["metrics"].get("within_blocker", "no probe")}
    s = [P.score(pr, x.float().numpy()) for x in f["feats"][f["n_pair"]:]]
    out = {}
    for st in ("p1", "p2", None):
        idx = [i for i, t in enumerate(test_tr) if st is None or t["set"] == st]
        neg = [s[i] for i in idx if test_tr[i]["ok"]]
        pos = [s[i] for i in idx if not test_tr[i]["ok"]]
        pp = P.pairs(idx, [not t["ok"] for t in test_tr], [t["pid"] for t in test_tr])
        sw, sc = [s[a] for a, _ in pp], [s[b] for _, b in pp]
        out[st or "pooled"] = {"problems": len({test_tr[i]["pid"] for i in idx}), "correct": len(neg), "wrong": len(pos),
                               "fpr": float(np.mean(np.array(neg) >= pr["tau"])) if neg else float("nan"),
                               "recall": float(np.mean(np.array(pos) >= pr["tau"])) if pos else float("nan"),
                               "pairs": len(pp), "within_auroc": P.pair_auroc(sw, sc), "within_ci": P.pair_ci(sw, sc)}
    return {"a4_tau": pr["tau"], "a4_pool": pr["pool"], "a4_layers": pr["name"], "a4_test": out}


def a3(c4, d):
    ok, deadline = c4.budget("A3")
    if not ok or not c4.has("A2", "pairs"):
        return {"A3_skipped": SKIP if not ok else "skipped: no contested pairs"}
    wrong = [t for t in c4.load("A2", "pairs") if not t["ok"]]
    sel = random.Random(0).sample(wrong, min(A3_MAX, len(wrong)))
    t0 = time.time()
    mc, n_rows, n_tok = mc_success(c4.s, sel, c4.run_id, "a3", deadline)
    first = {p: next((j for j, v in enumerate(v_) if v < DROP), None) for p, v_ in mc.items()}
    c4.s.save(d, "oracle", {p: {"mc": mc[p], "first_drop": first[p]} for p in mc})
    hist = collections.Counter(first.values())
    return {"a3_traces": len(mc), "a3_of": len(sel), "a3_completions": n_rows,
            "a3_tok_per_s": n_tok / max(time.time() - t0, 1e-9),
            "a3_restart_share": sum(j is not None and j <= 1 for j in first.values()) / len(first) if first else float("nan"),
            "a3_never_share": sum(j is None for j in first.values()) / len(first) if first else float("nan"),
            "a3_mc_prompt_mean": statistics.mean(v[0] for v in mc.values()) if mc else float("nan"),
            "a3_first_drop_hist": {**{str(j): hist[j] for j in sorted(k for k in hist if k is not None)},
                                   "never": hist[None]}}


STAGES = [("A1", a1), ("A2", a2), ("A4", a4), ("A3", a3)]
