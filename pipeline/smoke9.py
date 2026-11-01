"""SMOKE-9 (Qwen3.5-9B at cap 3200; one pod, <= 150 min, $6). No hint arms. Reuses the SMOKE-8 run on the volume
(config smoke8_source_run): Z2 problems / test units / harm traces / greedy test, Z4 train units, dev split, round-1
continuations and the rkl adapter.

Y0  environment (load, versions, fast path).
Y1  re-evaluation at cap 3200 with the SMOKE-8 rkl adapter (no training): base vs A6 greedy pass@1 on the 240 RC3 test
    problems, E0 vs E6 on the 47 test units, harm on the 60 correct traces. RC4 calibration at 3200 (60 problems,
    student greedy; the SMOKE-8 calibration problems): in [35%, 70%] -> RC4 becomes a second set for everything below.
Y2  RC4 data when added: 4 samples (cap 3200) for 200 train and 120 test problems, greedy test; contested; units; harm.
Y3  GRPO adapter on the SMOKE-8 training units (+ RC4): round 1 reuses the cached round-1 continuations as the first
    group (RC4 units sample theirs); reward = primary correctness; groups of 4 per prompt; DAPO dynamic sampling (groups
    all-correct or all-wrong dropped); A = (r - mean) / std; no KL; PPO clip 0.2 vs the sampling log-probs; lr 5e-5; dev
    every 10 steps on the SMOKE-8 dev traces (+ RC4 dev), early stopping across rounds, up to 3 rounds; kill/resume once.
Y4  the GRPO adapter evaluated exactly like Y1 (same seeds, paired).
Y5  report at cap 3200 with the 800-token numbers of SMOKE-8 side by side; brevity line = SMOKE-8 delta at 800 minus the
    same adapter's delta at 3200; FULL projection at cap 3200."""
import collections
import json
import pathlib
import random
import statistics
import time

import oracle as O
import smoke8 as S8
from core import seed_for
from smoke8 import Ctx8, _rate, contested, gen8, gen8_batches, logprobs8

CAP = 3200
RC4_TRAIN, RC4_TEST, HARM_N, CONT_K, ROUNDS = 200, 120, 60, 4, 3
LR, CLIP = 5e-5, 0.2
NS = "optE9"
BUDGET_MIN = {"Y0": 5, "Y1": 32, "Y2": 28, "Y3": 40, "Y4": 16, "Y5": 2}
START_MIN = {"Y0": 2, "Y1": 6, "Y2": 8, "Y3": 10, "Y4": 6, "Y5": 1}
ORDER = ["Y0", "Y1", "Y2", "Y3", "Y4", "Y5"]
SKIP = "skipped: not enough pod time left"


def grpo_groups(examples, k=CONT_K):
    """Group examples by pid (k per prompt); DAPO: drop groups whose rewards are all equal; advantage = (r - mean) / std
    per group (population std). Returns (kept examples with "adv", {groups, kept, dropped_all_correct, dropped_all_wrong})."""
    by = collections.defaultdict(list)
    for e in examples:
        by[e["pid"]].append(e)
    kept, stats = [], {"groups": len(by), "kept": 0, "dropped_all_correct": 0, "dropped_all_wrong": 0, "incomplete": 0}
    for pid, es in by.items():
        if len(es) < 2:
            stats["incomplete"] += 1
            continue
        r = [float(e["ok"]) for e in es]
        mean = sum(r) / len(r)
        std = (sum((x - mean) ** 2 for x in r) / len(r)) ** 0.5
        if std == 0:
            stats["dropped_all_correct" if r[0] == 1 else "dropped_all_wrong"] += 1
            continue
        stats["kept"] += 1
        kept += [{**e, "adv": (x - mean) / std} for e, x in zip(es, r)]
    return kept, stats


def brevity(delta_800, delta_3200):
    """How much of a delta measured at cap 800 disappears at cap 3200 (points)."""
    if delta_800 is None or delta_3200 is None:
        return None
    return delta_800 - delta_3200


class Ctx9(Ctx8):
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        super().__init__(cfg, tiers, tier, run_id, run_dir, max_new=CAP)
        self.lp_bs = {80: 4, 48: 2, 32: 1, 24: 1}[tier]            # logits [bs, 3300, 248k] in bf16
        self.micro = 2 if tier == 80 else 1
        self.src = pathlib.Path("/workspace/runs") / cfg["smoke8_source_run"]
        self.src_metrics = json.loads((self.src / "metrics.json").read_text()) if (self.src / "metrics.json").exists() else {}
        self.chosen8 = self.src_load("Z1", "chosen")
        self.sets = list(self.chosen8)
        S8.select_problems(self, self.chosen8)                     # SMOKE-8's RC3 problems, same seeds
        if (run_dir / "stage_Y1" / "rc4.pt").exists():            # a restart after Y1 decided on RC4
            self.sets = self.sets + (["rc4"] if self.load("Y1", "rc4") else [])
        self.rc4_probs()

    def src_load(self, stage, name):
        import torch
        return torch.load(self.src / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def rc4_probs(self):
        """RC4 problems (registered): calibration 60 (SMOKE-8 seed), train 200, test 120."""
        cal = self.register(self.rc_pool(4, S8.CAL_N, S8.RC_SEEDS["cal"] + 4))
        tr = [dict(p, split="train") for p in self.register(self.rc_pool(4, RC4_TRAIN, S8.RC_SEEDS["train"] + 4))]
        te = [dict(p, split="test") for p in self.register(self.rc_pool(4, RC4_TEST, S8.RC_SEEDS["test"] + 4))]
        return cal, tr, te

    def budget(self, name):
        from smoke3 import pipeline_deadline
        later = ORDER[ORDER.index(name) + 1:]
        end = pipeline_deadline() - sum(BUDGET_MIN[r] for r in later) * 60
        return time.time() + START_MIN[name] * 60 <= end, min(time.time() + BUDGET_MIN[name] * 60, end)

    def lora_init(self):
        return self.src_load("Z4", "lora_init")                     # the SMOKE-8 init, so its adapter loads unchanged

    def lora(self):
        import torch
        from train_repair import set_lora
        if not self._lora:
            from peft import LoraConfig, get_peft_model
            m = self.model()
            torch.manual_seed(seed_for(self.src_metrics.get("run_id", "SMOKE8"), "lora", 0))
            self._model = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                       target_modules=S8.lora_targets(m)))
            set_lora(self._model, self.lora_init())
            self._lora = True
        return self._model

    # SMOKE-8 (RC3) and Y2 (RC4) artifacts, per set
    def _per_set(self, name, s):
        if s == "rc4":
            return sorted(self.load("Y2", name), key=lambda t: t["pid"]) if self.has("Y2", name) else []
        return sorted(self.src_load("Z2", name), key=lambda t: t["pid"])

    def test_units(self, s):
        return self._per_set("test_units", s)

    def harm_traces(self, s):
        return self._per_set("harm", s)

    def greedy_test8(self, s):
        return self._per_set("greedy_test", s)


def y0(c9, d):
    import torch
    import transformers
    ok, _ = c9.budget("Y0")
    if not ok:
        return {"Y0_skipped": SKIP}
    t0 = time.time()
    model = c9.model()
    return {"y0_transformers": transformers.__version__, "y0_model_class": c9.model_class, "y0_load_sec": time.time() - t0,
            "y0_fast_path_warning": any("fast path" in w.lower() for w in c9.load_warnings), "y0_load_warnings": c9.load_warnings,
            "y0_vram_gb_after_load": torch.cuda.memory_allocated() / 2**30, "y0_cap": CAP, "y0_sets_from_smoke8": c9.chosen8,
            "y0_smoke8_run": c9.cfg["smoke8_source_run"], "y0_lp_bs": c9.lp_bs, "y0_micro": c9.micro}


def _pass1_stats(rows, groups):
    point, ci = O.cluster_bootstrap(rows, lambda xs: O.delta(xs, "a6", "base"), groups)
    _, ci_a6 = O.cluster_bootstrap(rows, lambda xs: O.recovery(xs, "a6"), groups, n=4000, seed=1)
    _, ci_base = O.cluster_bootstrap(rows, lambda xs: O.recovery(xs, "base"), groups, n=4000, seed=2)
    return {"n": len(rows), "base": O.recovery(rows, "base"), "base_ci": ci_base, "a6": O.recovery(rows, "a6"), "a6_ci": ci_a6,
            "delta": point, "delta_ci": ci, "base_tokens": _rate([r["base_tok"] for r in rows]), "a6_tokens": _rate([r["a6_tok"] for r in rows]),
            "base_trunc": _rate([r["base_trunc"] for r in rows]), "a6_trunc": _rate([r["a6_trunc"] for r in rows])}


def _arms_stats(rows):
    gs = [r["group"] for r in rows]
    point, ci = O.cluster_bootstrap(rows, lambda xs: O.delta(xs, "E6", "E0"), gs)
    return {"n": len(rows), "E0": O.recovery(rows, "E0"), "E6": O.recovery(rows, "E6"), "delta": point, "delta_ci": ci,
            "tok_E0": _rate([r["tok_E0"] for r in rows]), "tok_E6": _rate([r["tok_E6"] for r in rows]),
            "trunc_E0": _rate([r["trunc_E0"] for r in rows]), "trunc_E6": _rate([r["trunc_E6"] for r in rows])}


def evaluate_adapter(c9, d, state, tag, deadline, sets):
    """Cap-3200 evaluation of one adapter state on the given sets, one set at a time with set-scoped seeds (so a set's
    numbers pair across adapters and do not depend on which other sets exist): base vs A6 greedy pass@1 on the test
    problems, E0 vs E6 on the test units, harm on the correct traces. Rows per set are saved; pooled statistics are
    recomputed from every saved set of this tag."""
    from train_repair import set_lora
    model = c9.lora()
    m = {}
    for s in sets:
        base8 = c9.greedy_test8(s)
        mk = c9.mark()
        toks, trunc = c9.gen([t["prompt"] for t in base8], [0] * len(base8), deadline, sample=False)   # cache: 800-finished rows hit
        m.update(c9.since(mk, f"{tag}_base_{s}"))
        base = {t["pid"]: (tk, tr) for t, tk, tr in zip(base8, toks, trunc) if tk is not None}
        set_lora(model, state)
        t0 = time.time()
        a6_t, a6_tr = gen8_batches(model, [t["prompt"] for t in base8], c9.gen_bs, CAP, c9.eos, c9.pad, deadline=deadline, adapter_on=True)
        m[f"{tag}_a6_tok_per_s_{s}"] = sum(len(x) for x in a6_t) / max(time.time() - t0, 1e-9)
        rows = []
        for t, tk, tr in zip(base8, a6_t, a6_tr):
            if t["pid"] not in base:
                continue
            btk, btr = base[t["pid"]]
            rows.append({"pid": t["pid"], "set": t["set"], "group": c9.group(t["pid"]), "base": c9.ok(t["pid"], btk, btr),
                         "a6": c9.ok(t["pid"], tk, tr), "base_tok": len(btk), "a6_tok": len(tk), "base_trunc": btr, "a6_trunc": tr})
        c9.save(d, f"{tag}_pass1_{s}", rows)
        m[f"{tag}_pass1_done_{s}"] = f"{len(rows)}/{len(base8)}"
        if rows:
            m[f"{tag}_pass1_{s}"] = _pass1_stats(rows, [r["group"] for r in rows])
        # E0 vs E6 on the set's test units
        units = c9.test_units(s)
        seeds = O.chunk_seeds(c9.run_id, f"arms-{s}", len(units), c9.gen_bs)
        arms = {}
        for arm, on in (("E0", False), ("E6", True)):
            if time.time() > deadline:
                m[f"{tag}_arms_cut"] = f"deadline before {arm} on {s}"
                break
            toks, trunc = gen8_batches(model, [t["prompt"] for t in units], c9.gen_bs, CAP, c9.eos, c9.pad,
                                       seed_fn=lambda c: seeds[c], deadline=deadline, sample=True, adapter_on=on)
            if len(toks) < len(units):
                m[f"{tag}_arms_cut"] = f"deadline inside {arm} on {s}"
                break
            arms[arm] = [(c9.ok(t["pid"], tk, tr), len(tk), tr) for t, tk, tr in zip(units, toks, trunc)]
        if units and all(a in arms for a in ("E0", "E6")):
            rws = [{"pid": t["pid"], "set": t["set"], "group": c9.group(t["pid"]), "E0": arms["E0"][i][0], "E6": arms["E6"][i][0],
                    "tok_E0": arms["E0"][i][1], "tok_E6": arms["E6"][i][1], "trunc_E0": arms["E0"][i][2], "trunc_E6": arms["E6"][i][2]}
                   for i, t in enumerate(units)]
            c9.save(d, f"{tag}_arms_{s}", rws)
            m[f"{tag}_arms_{s}"] = _arms_stats(rws)
        # harm on the set's correct traces
        harm_tr = c9.harm_traces(s)
        hseeds = O.chunk_seeds(c9.run_id, f"harm-{s}", len(harm_tr), c9.gen_bs)
        for name, on in (("adapter", True), ("plain", False)):
            if time.time() > deadline or not harm_tr:
                break
            toks, trunc = gen8_batches(model, [t["prompt"] for t in harm_tr], c9.gen_bs, CAP, c9.eos, c9.pad, seed_fn=lambda c: hseeds[c],
                                       deadline=deadline, sample=True, adapter_on=on)
            if len(toks) == len(harm_tr):
                h = [not c9.ok(t["pid"], tk, tr) for t, tk, tr in zip(harm_tr, toks, trunc)]
                c9.save(d, f"{tag}_harm_{name}_{s}", h)
                m[f"{tag}_harm_{name}_{s}"] = _rate(h)
    # pooled over every set this tag has rows for (this stage dir and, for rkl, Y1's)
    allrows, allarms, harm = [], [], {"adapter": [], "plain": []}
    for st in ("rc3", "rc4"):
        for stage in ("Y1", "Y4"):
            if c9.has(stage, f"{tag}_pass1_{st}") and not any(r["set"] == st for r in allrows):
                allrows += c9.load(stage, f"{tag}_pass1_{st}")
            if c9.has(stage, f"{tag}_arms_{st}") and not any(r["set"] == st for r in allarms):
                allarms += c9.load(stage, f"{tag}_arms_{st}")
            for name in harm:
                if c9.has(stage, f"{tag}_harm_{name}_{st}") and st not in [x[1] for x in harm[name]]:
                    harm[name] += [(h, st) for h in c9.load(stage, f"{tag}_harm_{name}_{st}")]
    if allrows:
        m[f"{tag}_pass1_pooled"] = _pass1_stats(allrows, [r["group"] for r in allrows])
        m[f"{tag}_pass1_n"], m[f"{tag}_pass1_of"] = len(allrows), sum(len(c9.greedy_test8(s)) for s in sets)
    if allarms:
        m[f"{tag}_arms_pooled"] = _arms_stats(allarms)
    for name in harm:
        if harm[name]:
            m[f"{tag}_harm_{name}"] = _rate([h for h, _ in harm[name]])
    m[f"{tag}_harm_n"] = len(harm["plain"]) or len(harm["adapter"])
    m[f"{tag}_a6_tok_per_s"] = _rate([m[k] for k in m if k.startswith(f"{tag}_a6_tok_per_s_")])
    m[f"{tag}_base_tok_per_s"] = _rate([m[k] for k in m if k.startswith(f"{tag}_base_") and k.endswith("_tok_per_s")])
    return m


def y1(c9, d):
    ok, deadline = c9.budget("Y1")
    if not ok:
        return {"Y1_skipped": SKIP}
    if (c9.src / "stage_Z4" / "adapter.pt").exists():
        state, m = c9.src_load("Z4", "adapter"), {}
    else:
        state, m = c9.lora_init(), {"rkl_untrained": True}
    m.update(evaluate_adapter(c9, d, state, "rkl", deadline, list(c9.chosen8)))
    # RC4 calibration at 3200
    cal, _, _ = c9.rc4_probs()
    mk = c9.mark()
    toks, trunc = c9.gen([c9.student_prompt(p["pid"]) for p in cal], [0] * len(cal), deadline, sample=False)
    m.update(c9.since(mk, "y1_rc4cal"))
    oks = [c9.ok(p["pid"], tk, tr) for p, tk, tr in zip(cal, toks, trunc) if tk is not None]
    acc = _rate(oks) if len(oks) >= S8.CAL_N // 2 else None
    add = acc is not None and S8.CAL_LO <= acc <= S8.CAL_HI
    m.update({"y1_rc4_cal_n": len(oks), "y1_rc4_cal_acc": acc, "y1_rc4_cal_trunc": _rate([tr for tk, tr in zip(toks, trunc) if tk is not None]),
              "y1_rc4_cal_tokens": _rate([len(tk) for tk in toks if tk is not None]), "y1_rc4_added": add})
    c9.save(d, "rc4", add)
    if add and "rc4" not in c9.sets:
        c9.sets = c9.sets + ["rc4"]
    return m


def y2(c9, d):
    """RC4 data at cap 3200 when RC4 is in the band."""
    ok, deadline = c9.budget("Y2")
    if not c9.has("Y1", "rc4") or not c9.load("Y1", "rc4"):
        return {"Y2_skipped": "not needed: RC4 not in the accuracy band at 3200"}
    if not ok:
        return {"Y2_skipped": SKIP}
    _, tr, te = c9.rc4_probs()
    probs = random.Random(9).sample(tr + te, len(tr) + len(te))
    split = {p["pid"]: p["split"] for p in probs}
    rows = [(p["pid"], "g") for p in probs if p["split"] == "test"] + [(p["pid"], k) for p in probs for k in range(4)]
    greedy, sampled = [r for r in rows if r[1] == "g"], [r for r in rows if r[1] != "g"]
    mk = c9.mark()
    out = {}
    toks, trunc = c9.gen([c9.student_prompt(p) for p, _ in greedy], [0] * len(greedy), deadline, sample=False)
    out.update({r: (t, tr_) for r, t, tr_ in zip(greedy, toks, trunc) if t is not None})
    toks, trunc = c9.gen([c9.student_prompt(p) for p, _ in sampled], [seed_for(NS, "s", f"{p}#{k}") for p, k in sampled], deadline)
    out.update({r: (t, tr_) for r, t, tr_ in zip(sampled, toks, trunc) if t is not None})
    m = {**c9.since(mk, "y2"), "y2_rows_done": f"{len(out)}/{len(rows)}"}
    if len(out) < len(rows):
        m["y2_cut"] = f"deadline: {len(rows) - len(out)} of {len(rows)} rows not generated"
    traces = [dict(c9.trace(p, c9.student_prompt(p), *out[(p, k)], k=k), split=split[p])
              for p in dict.fromkeys(r[0] for r in rows) if all((p, j) in out for j in range(4)) for k in range(4)]
    gtest = [dict(c9.trace(p, c9.student_prompt(p), *out[(p, "g")]), split="test") for p in dict.fromkeys(r[0] for r in greedy) if (p, "g") in out]
    by = collections.defaultdict(list)
    for t in traces:
        by[t["pid"]].append(t)
    cont = {p for p, ts in by.items() if contested(ts)}
    gw = {t["pid"]: t for t in gtest}
    units = sorted([gw[p] for p in cont if split[p] == "test" and p in gw and not gw[p]["ok"] and gw[p]["gen"]], key=lambda t: t["pid"])
    correct = sorted([t for t in gtest if t["ok"]], key=lambda t: t["pid"])
    harm = sorted(random.Random(5).sample(correct, min(HARM_N, len(correct))), key=lambda t: t["pid"])
    c9.save(d, "samples", traces)
    c9.save(d, "greedy_test", gtest)
    c9.save(d, "test_units", units)
    c9.save(d, "harm", harm)
    c9.save(d, "train_units", sorted([{"pid": p, "set": "rc4", "group": c9.group(p)} for p in cont if split[p] == "train"], key=lambda u: u["pid"]))
    for sp in ("train", "test"):
        ps = [p for p in by if split[p] == sp]
        hist = collections.Counter(sum(t["ok"] for t in by[p]) for p in ps)
        m[f"y2_{sp}_rc4"] = {"problems": len(ps), "hist": [hist.get(k, 0) for k in range(5)], "acc": _rate([t["ok"] for p in ps for t in by[p]]),
                             "contested": _rate([p in cont for p in ps]), "trunc": _rate([t["truncated"] for p in ps for t in by[p]]),
                             **({"greedy_acc": _rate([gw[p]["ok"] for p in ps if p in gw]), "wrong_greedy_contested": len(units)} if sp == "test" else {})}
    m.update({"y2_problems": len(by), "y2_of": len(probs), "y2_mean_tokens": _rate([len(t["gen"]) for t in traces]),
              "y2_test_units": len(units), "y2_harm": len(harm)})
    return m


def _examples_from_smoke8(c9):
    """SMOKE-8 round-1 continuations (sampled at cap 800 with the untrained adapter) as the first GRPO group; the prefix
    is rebuilt from the problem id."""
    ex = []
    for e in c9.src_load("Z4", "examples_r1"):
        ex.append({**e, "prefix": c9.student_prompt(e["pid"]), "truncated": e.get("truncated", False)})
    return ex


def _sample_group(c9, units, tag, deadline, m):
    """CONT_K continuations per unit at cap 3200 with the current adapter; lp_s under that adapter."""
    model = c9.lora()
    rows = [(u, j) for u in units for j in range(CONT_K)]
    seqs = [c9.student_prompt(u["pid"]) for u, _ in rows]
    seeds = O.chunk_seeds(c9.run_id, f"cont{tag}", len(rows), c9.gen_bs)
    t0 = time.time()
    toks, trunc = c9.gen_adapter(seqs, seeds, deadline)
    gen_sec = time.time() - t0
    rows, seqs = rows[:len(toks)], seqs[:len(toks)]
    ex = [{"prefix": s, "cont": tk, "pid": u["pid"], "set": u["set"], "ok": c9.ok(u["pid"], tk, tr), "truncated": tr}
          for (u, _), s, tk, tr in zip(rows, seqs, toks, trunc) if tk]
    t1 = time.time()
    lp_s = logprobs8(model, [e["prefix"] for e in ex], [e["cont"] for e in ex], c9.lp_bs, adapter_on=True)
    for e, s in zip(ex, lp_s):
        e["lp_s"] = s
    lp_sec = time.time() - t1
    m.update({f"{tag}_units": len(units), f"{tag}_continuations": len(ex), f"{tag}_cont_primary": _rate([e["ok"] for e in ex]),
              f"{tag}_cont_trunc": _rate([e["truncated"] for e in ex]), f"{tag}_gen_sec": gen_sec, f"{tag}_lp_sec": lp_sec,
              f"{tag}_lp_tok_per_s": sum(len(e["prefix"]) + len(e["cont"]) for e in ex) / max(lp_sec, 1e-9),
              f"{tag}_gen_tok_per_s": sum(len(e["cont"]) for e in ex) / max(gen_sec, 1e-9),
              f"{tag}_mean_tokens": _rate([len(e["cont"]) for e in ex])})
    return ex


def y3(c9, d):
    import math
    import shutil
    import torch
    from core import kill_plan
    from train_repair import _loss_plain, set_lora, train
    ok, deadline = c9.budget("Y3")
    if not ok:
        return {"Y3_skipped": SKIP}
    units8, dev8 = c9.src_load("Z4", "train_units"), c9.src_load("Z4", "dev")
    units, dev = list(units8), list(dev8)
    if "rc4" in c9.sets and c9.has("Y2", "train_units"):
        dev4, tr4 = S8.dev_split(c9.load("Y2", "train_units"), n=15)
        units, dev = units + tr4, dev + dev4
    units = sorted(units, key=lambda u: u["pid"])
    model = c9.lora()
    set_lora(model, c9.lora_init())
    m = {"y3_units": len(units), "y3_units_by_set": dict(collections.Counter(u["set"] for u in units)), "y3_dev": len(dev),
         "y3_dev_by_set": dict(collections.Counter(u["set"] for u in dev)), "y3_lr": LR, "y3_clip": CLIP, "y3_micro": c9.micro}
    eval_fn = S8._dev_fn(c9, dev)
    best, curves, t0, eval_sec, gen_sec, executed = None, [], time.time(), 0.0, 0.0, 0
    for rnd in range(1, ROUNDS + 1):
        if rnd > 1 and time.time() + 300 > deadline:
            m[f"round{rnd}_skipped"] = "stage time budget"
            break
        if rnd > 1:
            if best is None:
                m[f"round{rnd}_skipped"] = "round 1 produced no checkpoint"
                break
            set_lora(model, best["state"])
        t1 = time.time()
        if rnd == 1:
            ex8 = _examples_from_smoke8(c9)                      # SMOKE-8 round-1 continuations (cap 800, base policy)
            ex = [e for e in ex8 if not e["truncated"]]          # a cap-800 truncation is not a cap-3200 reward
            m["y3r1_reused_smoke8"], m["y3r1_dropped_truncated"] = len(ex), len(ex8) - len(ex)
            rc4_units = [u for u in units if u["set"] == "rc4"]
            if rc4_units:
                ex += _sample_group(c9, rc4_units, "y3r1rc4", deadline, m)
        else:
            ex = _sample_group(c9, units, f"y3r{rnd}", deadline, m)
        gen_sec += time.time() - t1
        data, gstats = grpo_groups(ex)
        m[f"y3r{rnd}_groups"] = gstats
        m[f"y3r{rnd}_examples"] = len(data)
        c9.save(d, f"examples_r{rnd}", [{k: v for k, v in e.items() if k != "prefix"} for e in data])
        if not data or time.time() + 120 > deadline:
            m[f"round{rnd}_skipped"] = "no groups with mixed rewards" if not data else "stage time budget (sampled, training skipped)"
            break
        ck = d / f"ckpt_r{rnd}"
        shutil.rmtree(ck, ignore_errors=True)
        steps = math.ceil(len(data) / 16)
        early = {}
        kw = dict(loss_kind="grpo", lr=LR, clip_eps=CLIP, eval_fn=eval_fn, early=early, deadline=deadline - 60,
                  loss_fn=_loss_plain, micro=c9.micro)
        plan = kill_plan(steps) if rnd == 1 else None
        if plan:
            c, k = plan
            first, _ = train(model, data, ck, c9.run_id, stop_at=k, ckpt_every=c, **kw)
            set_lora(model, c9.lora_init())
            losses, _ = train(model, data, ck, c9.run_id, resume=True, ckpt_every=c, **kw)
            gap = [abs(first[s] - losses[s]) / max(1.0, abs(first[s])) for s in range(c + 1, k + 1) if s in losses and s in first]
            m.update({"train_ckpt_every": c, "train_kill_at": k, "resume_max_rel_diff": max(gap, default=float("nan")),
                      "resume_ok": bool(gap) and max(gap) <= 1e-3, "resume_exact": bool(gap) and max(gap) == 0.0})
            executed += k
        else:
            losses, _ = train(model, data, ck, c9.run_id, ckpt_every=10, **kw)
            if rnd == 1:
                m.update({"resume_ok": None, "train_blocker": f"only {steps} steps: kill/resume check needs >= 5"})
        eval_sec += early.get("eval_sec", 0.0)
        executed += early.get("last_step", 0) - (c if plan else 0)
        curves.append({"round": rnd, "steps": steps, "examples": len(data), "curve": early.get("curve", []), "best_step": early.get("best_step"),
                       "best_dev": early.get("best_score"), "stopped": early.get("stopped"), "cut": early.get("cut", False),
                       "loss_start": losses.get(1), "loss_end": losses.get(max(losses)) if losses else None})
        improved = early.get("best_state") is not None and (best is None or early["best_score"] > best["dev"])
        if improved:
            best = {"dev": early["best_score"], "step": early.get("best_step"), "round": rnd, "state": early["best_state"]}
        elif best is not None:
            m["rounds_stopped"] = f"round {rnd} did not improve dev"
            break
    sec = time.time() - t0
    if best is None:
        m["train_blocker"] = m.get("train_blocker") or "no round produced a checkpoint"
        set_lora(model, c9.lora_init())
    else:
        torch.save(best["state"], d / "adapter.pt")
        set_lora(model, best["state"])
    pure = max(sec - eval_sec - gen_sec, 1e-9)
    m.update({"train_rounds": curves, "train_rounds_done": len(curves), "train_best_dev": (best or {}).get("dev"),
              "train_best_step": (best or {}).get("step"), "train_best_round": (best or {}).get("round"),
              "early_stop_fired": any(c["stopped"] for c in curves), "train_sec": sec, "dev_eval_sec": eval_sec, "resample_sec": gen_sec,
              "executed_steps": executed, "train_steps_per_s": executed / max(sec, 1e-9), "pure_train_steps_per_s": executed / pure,
              "adapter_params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return m


def y4(c9, d):
    ok, deadline = c9.budget("Y4")
    if not ok:
        return {"Y4_skipped": SKIP}
    m = {}
    if "rc4" in c9.sets and not c9.has("Y1", "rkl_pass1_rc4"):       # the rkl adapter on the set added after Y1 (paired seeds)
        state = c9.src_load("Z4", "adapter") if (c9.src / "stage_Z4" / "adapter.pt").exists() else c9.lora_init()
        m.update(evaluate_adapter(c9, d, state, "rkl", deadline, ["rc4"]))
    if not c9.has("Y3", "adapter"):
        return {**m, "Y4_skipped": "skipped: no GRPO adapter"}
    m.update(evaluate_adapter(c9, d, c9.load("Y3", "adapter"), "grpo", deadline, list(c9.sets)))
    return m


def y5(c9, d):
    import report
    report.smoke9(c9.run_dir, c9.metrics, c9.cfg, c9.src_metrics)
    return {}


STAGES = [("Y0", y0), ("Y1", y1), ("Y2", y2), ("Y3", y3), ("Y4", y4), ("Y5", y5)]
