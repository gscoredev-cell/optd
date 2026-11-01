"""SMOKE-6 / SMOKE-7 (OPTION_E_SPEC.md §6 and the SMOKE-7 addendum): trained repair under an oracle pointer. Stages in
the deadline order [E-S0 data,] E-S1, E-S2, E-S4, E-S5, E-S6, E-S7, E-S8 (report), then E-S3 (bisection validation,
optional; the report is rewritten once more after it). Every stage has a time budget and reserves the budgets of the
required stages after it; a stage that cannot start in time is skipped and says so.

Data: SMOKE-4 A1 traces (400 train + 400 test problems x 4 samples, Qwen3-1.7B, T 0.8, cap 800) define the contested
stratum (1..3 of 4 correct) and are imported into the generation cache; A3's 48 full MC scans are imported as MC
vectors (the completions were not stored). SMOKE-7 adds E-S0: 4 samples for 800 more train problems (p1 400, p2 200,
RC3 200) and 500 more test problems (p1/p2), through the cache. First attempts are greedy (through the cache). Train
units = wrong greedy traces of contested train problems (bisection) + the A3-scanned sampled wrong traces (own scans,
§4; SMOKE-6 only, since A3's MC rule differs from SMOKE-7's) + sampled wrong traces whose first b* boundaries share
>= 70% of their tokens with the greedy trace (reuse its cut). Test units = wrong greedy traces of contested test
problems. Adapter-on generation never goes through the cache (the adapter changes it). torch is imported lazily so the
pure parts (arm construction, statistics) are testable without it."""
import collections
import pathlib
import random
import statistics
import time

import oracle as O
from core import primary_ok, seed_for
from gencache import GenCache, cached_decode

MAX_NEW = 800
INJECT = "\nWait — let me re-check that step.\n"
E5_N, VALID_N, IDENT_N, CONT_K, SLOPPY, OVERLAP = 60, 20, 20, 4, 0.25, 0.70
SKIP = "skipped: not enough pod time left"
BASE_ARMS = ("E0", "E1", "E2")
ADAPTER_ARMS = ("E3", "E4a", "E4b", "E6")
ARMS = BASE_ARMS + ADAPTER_ARMS

# variants: SMOKE-6 (spec v1.6 §6) and SMOKE-7 (addendum)
V6 = {"name": "SMOKE6", "k": 3, "thresh": 0.5, "lr": 1e-4, "clip": None, "dev_n": 60, "rounds": 2, "adapters": ("rkl",),
      "extra_data": False, "train_cap": 200, "test_cap": 120, "a3_units": True,
      # minutes: spec L40 estimates x 1.3 (E-S8 = report), and the least time a stage may start with
      "budget": {"ES1": 4, "ES2": 33, "ES4": 11, "ES5": 18, "ES6": 20, "ES7": 16, "ES8": 3, "ES3": 8},
      "start": {"ES1": 2, "ES2": 8, "ES4": 4, "ES5": 5, "ES6": 6, "ES7": 6, "ES8": 1, "ES3": 3}}
V7 = {"name": "SMOKE7", "k": 4, "thresh": 0.25, "lr": 5e-5, "clip": 0.2, "dev_n": 30, "rounds": 3, "adapters": ("rkl", "rft"),
      "extra_data": True, "train_cap": 300, "test_cap": 200, "a3_units": False,
      # minutes: A100 SMOKE-6 rates x 1.3; sum of required stages 103 of ~106 usable
      "budget": {"ES0": 28, "ES1": 1, "ES2": 12, "ES4": 8, "ES5": 26, "ES6": 12, "ES7": 14, "ES8": 2, "ES3": 6},
      "start": {"ES0": 6, "ES1": 1, "ES2": 4, "ES4": 3, "ES5": 5, "ES6": 4, "ES7": 5, "ES8": 1, "ES3": 2}}
VARIANTS = {"SMOKE6": V6, "SMOKE7": V7}
EXTRA_SIZES = {("train", "p1"): 400, ("train", "p2"): 200, ("train", "rc3"): 200, ("test", "p1"): 250, ("test", "p2"): 250}
RC3_SEED = 703        # used so far: 0, 103-105, 200, 503/504, 603/604


class Ctx6:
    def __init__(self, cfg, tiers, tier, run_id, run_dir, variant=V6):
        import data as D
        from smoke4 import Ctx4
        self.cfg, self.run_id, self.run_dir, self.v = cfg, run_id, run_dir, variant
        self.c4 = Ctx4(cfg, tiers, tier, run_id, run_dir)      # probs (800), split, group, gold; gen_bs x 4
        self.s = self.c4.s
        self.s.gen_bs = min(self.s.gen_bs, 256)              # 1.7B KV at cap 800: 256 rows ~ 47 GB (tier 80 only)
        self.src = pathlib.Path("/workspace/runs") / cfg["smoke4_source_run"]
        self.cache = GenCache(cfg["student"])
        self.store = O.MCStore(cfg["student"], k=variant["k"])
        self.inject_ids = self.s.tok.encode(INJECT, add_special_tokens=False)
        self.extra = []
        if variant["extra_data"]:                            # E-S0 problems, disjoint from SMOKE-4's 800
            gsm = Ctx4._gsm(cfg)
            oids = dict(zip(("train", "test"), D.make_split(gsm)))
            used = {p["pid"] for p in self.c4.probs}
            for (sp, st), n in EXTRA_SIZES.items():
                if st == "rc3":
                    ps = [dict(p, pid=f"rc3:{p['pid'].split(':', 1)[1]}", set="rc3")
                          for p in D.random_calculation(n, seed=RC3_SEED, min_ops=3, max_ops=3)]
                else:
                    ps = D.sample([p for p in D.problems(gsm, [st], oids[sp]) if p["pid"] not in used], n, seed=7)
                self.extra += [dict(p, split=sp) for p in ps]
            self.extra = random.Random(7).sample(self.extra, len(self.extra))   # a time cut keeps every set
            self.c4.split.update({p["pid"]: p["split"] for p in self.extra})
            self.c4.group.update({p["pid"]: f"gsm:{p['oid']}" if p["oid"] is not None else p["pid"] for p in self.extra})
            self.s.gold.update({p["pid"]: p["gold"] for p in self.extra})

    @property
    def order(self):
        return (["ES0"] if self.v["extra_data"] else []) + ["ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8"]

    # time
    def budget(self, name):
        """(start allowed, stage deadline); the deadline leaves every later required stage its nominal budget."""
        from smoke3 import pipeline_deadline
        order = self.order
        later = order[order.index(name) + 1:] if name in order else ["ES8"]
        end = pipeline_deadline() - sum(self.v["budget"][r] for r in later) * 60
        return time.time() + self.v["start"][name] * 60 <= end, min(time.time() + self.v["budget"][name] * 60, end)

    # artifacts
    def load(self, stage, name):
        import torch
        return torch.load(self.run_dir / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def has(self, stage, name):
        return (self.run_dir / f"stage_{stage}" / f"{name}.pt").exists()

    def src_load(self, stage, name):
        import torch
        return torch.load(self.src / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    # generation
    def gen(self, seqs, seeds, deadline, budgets=None, sample=True):
        """Adapter-off generation through the cache (the student may be LoRA-wrapped; decode disables the adapter)."""
        return cached_decode(self.cache, self.s.student(), seqs, seeds, self.s.gen_bs, MAX_NEW, self.s.eos, self.s.pad,
                             budgets=budgets, deadline=deadline, sample=sample)

    def mc_gen(self, deadline):
        """(gen, score) pair for oracle.probe_points."""
        def gen_fn(prefixes, seeds, cuts):
            return self.gen(prefixes, seeds, deadline, budgets=[MAX_NEW - c for c in cuts])

        def score(t, cut, toks, trunc):
            return primary_ok(self.s.tok.decode(t["gen"][:cut] + toks, skip_special_tokens=True),
                              self.s.gold[t["pid"]], trunc)
        return gen_fn, score

    def trace(self, pid, prompt, gen, trunc, k="g"):
        return dict(self.s.trace(pid, prompt, gen, trunc), k=k, split=self.c4.split[pid])

    def greedy(self, pids, prompts, deadline):
        """Greedy first attempts (cache) for the given problems; traces for those generated in time."""
        toks, trunc = self.gen([prompts[p] for p in pids], [0] * len(pids), deadline, sample=False)
        return [self.trace(p, prompts[p], tk, tr) for p, tk, tr in zip(pids, toks, trunc) if tk is not None]

    def mark(self):
        return self.cache.hits, self.cache.misses, self.cache.gen_tokens, time.time()

    def since(self, mark, tag):
        h, mi, tok, t0 = mark
        sec = max(time.time() - t0, 1e-9)
        return {f"{tag}_cache_hits": self.cache.hits - h, f"{tag}_cache_misses": self.cache.misses - mi,
                f"{tag}_tok_per_s": (self.cache.gen_tokens - tok) / sec, f"{tag}_sec": sec}

    def unit_ok(self, t, cut, new, trunc):
        return primary_ok(self.s.tok.decode(t["gen"][:cut] + new, skip_special_tokens=True), self.s.gold[t["pid"]],
                          trunc)

    def a1_traces(self):
        """SMOKE-4 A1 traces plus E-S0's (when it ran), all with split."""
        traces = [dict(t, split=self.c4.split[t["pid"]]) for t in self.src_load("A1", "traces")]
        if self.has("ES0", "traces"):
            traces += self.load("ES0", "traces")
        return traces


def by_problem(traces):
    out = collections.defaultdict(list)
    for t in traces:
        out[t["pid"]].append(t)
    return out


def contested(ts):
    return 0 < sum(t["ok"] for t in ts) < len(ts)


def token_share(a, b, n):
    """Share of the first n positions where a and b hold the same token (0 when n is 0)."""
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else 0.0


def per_set(units, key=lambda u: u["trace"]["set"]):
    out = collections.defaultdict(list)
    for u in units:
        out[key(u)].append(u)
        out["pooled"].append(u)
    return out


def oracle_summary(units, tag):
    """Restart share, b* histogram and recoverable-to-the-end counts per set and pooled."""
    m = {}
    for s, us in per_set(units).items():
        bs = [u["b"] for u in us]
        m[f"{tag}_{s}"] = {"n": len(us), "restart_share": sum(b == 0 for b in bs) / len(bs) if bs else float("nan"),
                           "to_end": sum(u["b"] == len(u["trace"]["bounds"]) and u["b"] > 0 for u in us),
                           "b_hist": {str(k): v for k, v in sorted(collections.Counter(bs).items())},
                           "b_ge1": sum(b >= 1 for b in bs)}
    return m


def es0(c6, d):
    """SMOKE-7 data: 4 samples (T 0.8, cap 800) per extra problem through the cache; contested counts per set."""
    from generate import prompt_ids
    ok, deadline = c6.budget("ES0")
    if not ok:
        return {"ES0_skipped": SKIP}
    prompts = {p["pid"]: prompt_ids(c6.s.tok, p["question"]) for p in c6.extra}
    rows = [(p["pid"], k) for p in c6.extra for k in range(4)]
    mk = c6.mark()
    toks, trunc = c6.gen([prompts[p] for p, _ in rows], [seed_for(O.NS, "a1", f"{p}#{k}") for p, k in rows], deadline)
    got = {r: (tk, tr) for r, tk, tr in zip(rows, toks, trunc) if tk is not None}
    traces = [c6.trace(p, prompts[p], *got[(p, k)], k=k) for p in dict.fromkeys(r[0] for r in rows)
              if all((p, j) in got for j in range(4)) for k in range(4)]
    c6.s.save(d, "traces", traces)
    m = {**c6.since(mk, "es0"), "es0_problems": len(traces) // 4, "es0_of": len(c6.extra), "es0_bs": c6.s.gen_bs,
         "es0_trunc": statistics.mean(t["truncated"] for t in traces) if traces else float("nan"),
         "es0_mean_tokens": statistics.mean(len(t["gen"]) for t in traces) if traces else float("nan")}
    for (sp, st), n in EXTRA_SIZES.items():
        ps = [ts for p, ts in by_problem(traces).items() if ts[0]["split"] == sp and ts[0]["set"] == st]
        hist = collections.Counter(sum(t["ok"] for t in ts) for ts in ps)
        m[f"es0_{sp}_{st}"] = {"problems": len(ps), "of": n, "hist": [hist.get(k, 0) for k in range(5)],
                               "acc": statistics.mean(t["ok"] for ts in ps for t in ts) if ps else float("nan"),
                               "contested": sum(map(contested, ps)) / len(ps) if ps else float("nan")}
    return m


def es1(c6, d):
    """Import SMOKE-4 A1 traces (cache) and A3 scans (MC store, k = 3 only); select contested problems; counts."""
    ok, _ = c6.budget("ES1")
    if not ok:
        return {"ES1_skipped": SKIP}
    traces = c6.src_load("A1", "traces")
    new = [t for t in traces if c6.cache.get(t["prompt"], 0.8, 0.95, seed_for(O.NS, "a1", O.sid(t)), MAX_NEW) is None]
    c6.cache.put_many([(t["prompt"], 0.8, 0.95, seed_for(O.NS, "a1", O.sid(t)), MAX_NEW, t["gen"], t["truncated"])
                       for t in new])
    pairs, orc = c6.src_load("A2", "pairs"), c6.src_load("A3", "oracle")
    wrong = {t["pid"]: t for t in pairs if not t["ok"]}
    a3 = [(wrong[p], v["mc"]) for p, v in orc.items() if p in wrong and len(v["mc"]) == len(wrong[p]["bounds"]) + 1]
    a3_new = c6.store.import_vectors(a3) if c6.v["k"] == O.MC_K else 0
    bp = by_problem(c6.a1_traces())
    cont = {p: ts for p, ts in bp.items() if contested(ts)}
    sel = [t for p in sorted(cont) for t in cont[p]]
    c6.s.save(d, "a1", sel)
    c6.s.save(d, "a3", [{"trace": dict(t, split=c6.c4.split[t["pid"]]), "mc": mc, "b": O.oracle_cut(mc, c6.v["thresh"])}
                        for t, mc in a3])
    m = {"es1_a1_traces": len(traces), "es1_a1_imported": len(new), "es1_a3_scans": len(a3), "es1_a3_points_new": a3_new,
         "es1_a3_bad_length": len(orc) - len(a3), "es1_all_traces": len(bp) * 4}
    for sp in ("train", "test"):
        ps = [p for p in cont if c6.c4.split[p] == sp]
        m[f"es1_contested_{sp}"] = {**{s: sum(p.split(":")[0] == s for p in ps) for s in ("p1", "p2", "main", "rc3")},
                                    "all": len(ps)}
    return m


def _oracle_stage(c6, d, split, cap, tag, deadline):
    """Greedy first attempts on the contested problems of `split`, then bisection on the wrong ones (<= cap)."""
    a1 = [t for t in c6.load("ES1", "a1") if t["split"] == split]
    prompts = {t["pid"]: t["prompt"] for t in a1}
    pids = sorted(prompts)
    mk = c6.mark()
    greedy = c6.greedy(pids, prompts, deadline)
    m = {**c6.since(mk, f"{tag}_greedy"), f"{tag}_contested": len(pids), f"{tag}_greedy": len(greedy),
         f"{tag}_greedy_wrong": sum(not t["ok"] for t in greedy),
         f"{tag}_greedy_acc": statistics.mean(t["ok"] for t in greedy) if greedy else float("nan")}
    c6.s.save(d, "greedy", greedy)
    wrong = [t for t in greedy if not t["ok"] and t["bounds"]]
    wrong = sorted(random.Random(6).sample(wrong, min(cap, len(wrong))), key=O.sid)
    mk = c6.mark()
    done, rounds = O.run_bisection(c6.mc_gen(deadline), c6.store, wrong, deadline, thresh=c6.v["thresh"], k=c6.v["k"])
    m.update(c6.since(mk, f"{tag}_mc"))
    units = [{"trace": t, "b": done[O.sid(t)]["b"], "probed": done[O.sid(t)]["probed"], "src": "bisect"}
             for t in wrong if O.sid(t) in done]
    m.update({f"{tag}_bisect_traces": len(units), f"{tag}_bisect_of": len(wrong), f"{tag}_bisect_rounds": rounds,
              f"{tag}_probes_per_trace": statistics.mean(len(u["probed"]) for u in units) if units else float("nan")})
    return greedy, units, m


def es2(c6, d):
    """Oracle labels, train: bisection on wrong greedy traces of contested train problems; A3 units reused (k = 3)."""
    ok, deadline = c6.budget("ES2")
    if not ok or not c6.has("ES1", "a1"):
        return {"ES2_skipped": SKIP if not ok else "skipped: no ES1 artifacts"}
    greedy, units, m = _oracle_stage(c6, d, "train", c6.v["train_cap"], "es2", deadline)
    by_g = {t["pid"]: t for t in greedy}
    seen = {O.sid(u["trace"]) for u in units}
    a3 = [dict(u, src="a3") for u in c6.load("ES1", "a3") if u["b"] is not None] if c6.v["a3_units"] else []
    a3 = [u for u in a3 if O.sid(u["trace"]) not in seen]
    seen |= {O.sid(u["trace"]) for u in a3}
    reuse, a1 = [], by_problem(c6.load("ES1", "a1"))
    for u in units:                      # sampled wrong traces of the same problem that share >= 70% of the prefix
        if u["b"] < 1:
            continue
        g = u["trace"]
        for t in a1.get(g["pid"], []):
            if not t["ok"] and len(t["bounds"]) >= u["b"] and O.sid(t) not in seen:
                n = O.prefix_len(t["bounds"], u["b"])        # the sampled trace's own prefix up to boundary b*
                if token_share(g["gen"], t["gen"], n) >= OVERLAP:
                    reuse.append({"trace": t, "b": u["b"], "probed": {}, "src": "reuse"})
                    seen.add(O.sid(t))
    units = units + a3 + reuse
    c6.s.save(d, "units", units)
    m.update({"es2_units": len(units), "es2_units_a3": len(a3), "es2_units_reuse": len(reuse),
              "es2_units_b_ge1": sum(u["b"] >= 1 for u in units), "es2_a3_greedy_wrong_overlap":
              sum(1 for u in a3 if u["trace"]["pid"] in by_g and not by_g[u["trace"]["pid"]]["ok"])})
    m.update(oracle_summary(units, "es2_oracle"))
    m.update(oracle_summary([u for u in units if u["src"] == "bisect"], "es2_oracle_greedy"))
    return m


def _round(c6, units, model, tag, deadline, m):
    """Continuations x CONT_K from each unit's prefix with the adapter on, scored by the teacher and by the sampling
    student (adapter on); teacher greedy from the same prefix for teacher-also-wrong. Returns training examples."""
    from generate import decode_batches, load_model, split_logprobs, token_logprobs
    from smoke import free
    rows = [(u, j) for u in units for j in range(CONT_K)]
    seqs = [u["trace"]["prompt"] + u["trace"]["gen"][:O.prefix_len(u["trace"]["bounds"], u["b"])] for u, _ in rows]
    cuts = [len(s) - len(u["trace"]["prompt"]) for s, (u, _) in zip(seqs, rows)]
    t0 = time.time()
    toks, trunc = decode_batches(model, seqs, c6.s.gen_bs, MAX_NEW, c6.s.eos, c6.s.pad,
                                 seed_fn=lambda c: seed_for(c6.run_id, f"cont{tag}", c),
                                 budgets=[MAX_NEW - c for c in cuts], sample=True, adapter_on=True, deadline=deadline)
    gen_sec = time.time() - t0
    rows, seqs, cuts = rows[:len(toks)], seqs[:len(toks)], cuts[:len(toks)]
    ex = [{"prefix": s, "cont": tk, "cut": c, "sid": O.sid(u["trace"]), "sloppy": u.get("sloppy", False),
           "ok": c6.unit_ok(u["trace"], c, tk, tr), "truncated": tr}
          for (u, _), s, tk, c, tr in zip(rows, seqs, toks, cuts, trunc) if tk]
    lp_s = split_logprobs(model, [e["prefix"] for e in ex], [e["cont"] for e in ex], c6.s.pre_bs)   # the sampler's policy
    teacher = load_model(c6.cfg["teacher"])
    t1 = time.time()
    lp_t = token_logprobs(teacher, [e["prefix"] for e in ex], [e["cont"] for e in ex], c6.s.pre_bs)
    prefill_sec = time.time() - t1
    tcache = GenCache(c6.cfg["teacher"])
    useqs = [u["trace"]["prompt"] + u["trace"]["gen"][:O.prefix_len(u["trace"]["bounds"], u["b"])] for u in units]
    ttoks, ttr = cached_decode(tcache, teacher, useqs, [0] * len(useqs), c6.s.pre_bs, MAX_NEW, c6.s.eos, c6.s.pad,
                               budgets=[MAX_NEW - (len(s) - len(u["trace"]["prompt"])) for s, u in zip(useqs, units)],
                               deadline=deadline, sample=False)
    twrong = [not c6.unit_ok(u["trace"], len(s) - len(u["trace"]["prompt"]), tk, tr)
              for u, s, tk, tr in zip(units, useqs, ttoks, ttr) if tk is not None]
    del teacher
    free()
    for e, s, t in zip(ex, lp_s, lp_t):
        e["lp_s"], e["lp_t"] = s, t
    adv = [float((-(e["lp_s"] - e["lp_t"])).clamp(-5, 5).mean()) for e in ex if len(e["cont"])]
    m.update({f"{tag}_units": len(units), f"{tag}_sloppy_units": sum(u.get("sloppy", False) for u in units),
              f"{tag}_continuations": len(ex), f"{tag}_cont_primary": statistics.mean(e["ok"] for e in ex) if ex else float("nan"),
              f"{tag}_cont_trunc": statistics.mean(e["truncated"] for e in ex) if ex else float("nan"),
              f"{tag}_gen_sec": gen_sec, f"{tag}_gen_tok_per_s": sum(len(e["cont"]) for e in ex) / max(gen_sec, 1e-9),
              f"{tag}_prefill_tok_per_s": sum(len(e["prefix"]) + len(e["cont"]) for e in ex) / max(prefill_sec, 1e-9),
              f"{tag}_teacher_also_wrong": statistics.mean(twrong) if twrong else float("nan"),
              f"{tag}_teacher_scored": len(twrong), f"{tag}_mean_adv": statistics.mean(adv) if adv else float("nan")})
    return ex


def _split_dev(units, n):
    """n dev traces (one unit each, seeded) never used for training; every unit of a dev trace's problem is kept out
    of training (dev is split by problem, like test). Pointed traces first, so the dev signal is about the pointer."""
    traces = sorted({O.sid(u["trace"]): u for u in units}.items())
    pointed = [x for x in traces if x[1]["b"] >= 1]
    rest = [x for x in traces if x[1]["b"] == 0]
    rng = random.Random(60)
    pick = rng.sample(pointed, min(n, len(pointed)))
    pick += rng.sample(rest, min(n - len(pick), len(rest)))
    dev_sids = {s for s, _ in pick}
    dev = [u for s, u in traces if s in dev_sids]
    dev_pids = {u["trace"]["pid"] for u in dev}
    return dev, [u for u in units if u["trace"]["pid"] not in dev_pids]


def es4(c6, d):
    """Round-1 continuations (+25% sloppy cuts) from the oracle prefixes with the untrained adapter; teacher scoring."""
    ok, deadline = c6.budget("ES4")
    if not ok or not c6.has("ES2", "units"):
        return {"ES4_skipped": SKIP if not ok else "skipped: no ES2 units"}
    dev, train_units = _split_dev(c6.load("ES2", "units"), c6.v["dev_n"])
    from train_repair import sloppy_mix
    train_units = sloppy_mix(train_units, SLOPPY, seed_for(c6.run_id, "sloppy", 0))
    c6.s.save(d, "dev", dev)
    c6.s.save(d, "train_units", train_units)
    model = c6.s.lora()
    m = {"es4_dev_traces": len(dev), "es4_dev_b_ge1": sum(u["b"] >= 1 for u in dev),
         "es4_train_units_b_ge1": sum(u["b"] >= 1 for u in train_units)}
    ex = _round(c6, train_units, model, "es4", deadline, m)
    c6.s.save(d, "examples", ex)
    return m


def _dev_eval(c6, dev):
    """E3 recovery on the dev traces with one continuation each and fixed seeds (comparable across steps)."""
    from eval import continue_from
    rows = [(u["trace"], O.prefix_len(u["trace"]["bounds"], u["b"]), []) for u in dev]
    seeds = O.chunk_seeds(c6.run_id, "dev", len(rows), c6.s.gen_bs)

    def fn(model):
        toks, trunc = continue_from(model, rows, c6.s.gen_bs, MAX_NEW, c6.s.eos, c6.s.pad, seeds, adapter_on=True)
        return statistics.mean(c6.unit_ok(t, cut, tk, tr) for (t, cut, _), tk, tr in zip(rows, toks, trunc))
    return fn


def _pct(x):
    return "nan" if x is None else f"{100 * x:.1f}%"


def _train_adapter(c6, d, kind, units, dev, ex1, deadline, m):
    """Train one adapter (rkl with optional PPO clipping, or rft = SFT on correct continuations) for up to v["rounds"]
    rounds; each round resamples continuations with the current adapter; the best dev checkpoint across rounds is
    kept and rounds stop once a round fails to improve it. Round 1 of the first adapter runs the kill/resume check."""
    import math
    import shutil
    import torch
    from core import kill_plan
    from train_repair import set_lora, train
    model = c6.s.lora()
    eval_fn = _dev_eval(c6, dev)
    set_lora(model, c6.s.lora_init())
    pre = f"{kind}_"
    best, curves, t0, eval_sec, gen_sec, executed = None, [], time.time(), 0.0, 0.0, 0
    ex = ex1
    for rnd in range(1, c6.v["rounds"] + 1):
        if rnd > 1:
            if time.time() + 240 > deadline:
                m[f"{pre}round{rnd}_skipped"] = "stage time budget"
                break
            set_lora(model, best["state"])
            t1 = time.time()
            ex = _round(c6, units, model, f"es5_{kind}_r{rnd}", deadline, m)
            gen_sec += time.time() - t1
            c6.s.save(d, f"examples_{kind}_r{rnd}", ex)
            if time.time() + 120 > deadline:
                m[f"{pre}round{rnd}_skipped"] = "stage time budget (continuations sampled, training skipped)"
                break
        data = [e for e in ex if e["ok"]] if kind == "rft" else ex
        m[f"{pre}train_examples_r{rnd}"] = len(data)
        if not data:
            m[f"{pre}train_blocker"] = f"no training examples in round {rnd}"
            break
        ck = d / f"ckpt_{kind}_r{rnd}"
        shutil.rmtree(ck, ignore_errors=True)
        steps = math.ceil(len(data) / 16)
        early = {}
        kw = dict(loss_kind=kind, lr=c6.v["lr"], clip_eps=c6.v["clip"] if kind == "rkl" else None, eval_fn=eval_fn,
                  early=early, deadline=deadline - 60)
        plan = kill_plan(steps) if rnd == 1 and kind == c6.v["adapters"][0] else None
        if plan:
            c, k = plan
            first, _ = train(model, data, ck, c6.run_id, stop_at=k, ckpt_every=c, **kw)
            set_lora(model, c6.s.lora_init())             # the "killed" process's weights are gone; resume restores them
            losses, _ = train(model, data, ck, c6.run_id, resume=True, ckpt_every=c, **kw)
            gap = [abs(first[s] - losses[s]) / max(1.0, abs(first[s])) for s in range(c + 1, k + 1)
                   if s in losses and s in first]
            executed += k
            m.update({"train_ckpt_every": c, "train_kill_at": k, "resume_max_rel_diff": max(gap, default=float("nan")),
                      "resume_ok": bool(gap) and max(gap) <= 1e-3, "resume_exact": bool(gap) and max(gap) == 0.0})
        else:
            losses, _ = train(model, data, ck, c6.run_id, ckpt_every=10, **kw)
            if rnd == 1 and kind == c6.v["adapters"][0]:
                m.update({"resume_ok": None, "train_blocker": f"only {steps} steps: kill/resume check needs >= 5"})
        eval_sec += early.get("eval_sec", 0.0)
        curves.append({"round": rnd, "steps": steps, "curve": early.get("curve", []), "best_step": early.get("best_step"),
                       "best_dev": early.get("best_score"), "stopped": early.get("stopped"),
                       "loss_start": losses.get(1), "loss_end": losses.get(max(losses)) if losses else None})
        m[f"{pre}train_steps_total"] = m.get(f"{pre}train_steps_total", 0) + early.get("last_step", 0)
        executed += early.get("last_step", 0) - (c if plan else 0)      # resumed run re-executed steps c+1..
        if early.get("cut"):
            m[f"{pre}round{rnd}_cut"] = f"training cut by the stage deadline at step {early.get('last_step')}"
        improved = early.get("best_state") is not None and (best is None or early["best_score"] > best["dev"])
        if improved:
            best = {"dev": early["best_score"], "step": early.get("best_step"), "round": rnd, "state": early["best_state"]}
        elif best is not None:
            m[f"{pre}rounds_stopped"] = f"round {rnd} did not improve dev ({_pct(early.get('best_score'))} <= {_pct(best['dev'])})"
            break
    sec = time.time() - t0
    if best is None:
        best = {"dev": None, "step": 0, "round": 0, "state": c6.s.lora_init()}
    torch.save(best["state"], d / f"adapter_{kind}.pt")
    steps_total = m.get(f"{pre}train_steps_total", 0)
    pure = max(sec - eval_sec - gen_sec, 1e-9)
    m.update({f"{pre}train_rounds": curves, f"{pre}train_rounds_done": len(curves), f"{pre}train_best_dev": best["dev"],
              f"{pre}train_best_step": best["step"], f"{pre}train_best_round": best["round"],
              f"{pre}early_stop_fired": any(c["stopped"] for c in curves), f"{pre}train_sec": sec,
              f"{pre}dev_eval_sec": eval_sec, f"{pre}resample_sec": gen_sec,
              f"{pre}executed_steps": executed, f"{pre}train_steps_per_s": executed / max(sec, 1e-9),
              f"{pre}pure_train_steps_per_s": executed / pure})
    return best


def es5(c6, d):
    """Train the adapter(s) with early stopping on the dev set; the first adapter's round 1 runs the kill/resume check."""
    from train_repair import set_lora
    ok, deadline = c6.budget("ES5")
    if not ok or not c6.has("ES4", "examples"):
        return {"ES5_skipped": SKIP if not ok else "skipped: no ES4 examples"}
    model, dev, units = c6.s.lora(), c6.load("ES4", "dev"), c6.load("ES4", "train_units")
    ex1 = c6.load("ES4", "examples")
    m = {"train_examples_r1": len(ex1), "adapter_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
         "train_lr": c6.v["lr"], "train_clip": c6.v["clip"], "train_adapters": list(c6.v["adapters"])}
    for kind in c6.v["adapters"]:
        if kind != c6.v["adapters"][0] and time.time() + 300 > deadline:
            m[f"{kind}_skipped"] = "stage time budget"
            continue
        _train_adapter(c6, d, kind, units, dev, ex1, deadline, m)
    first = c6.v["adapters"][0]                      # legacy keys (report / gates) mirror the first adapter
    for k in ("train_rounds", "train_rounds_done", "train_best_dev", "train_best_step", "train_best_round",
              "early_stop_fired", "train_steps_per_s", "train_sec", "train_steps_total", "pure_train_steps_per_s"):
        if f"{first}_{k}" in m:
            m[k] = m[f"{first}_{k}"]
    set_lora(model, c6.load("ES5", f"adapter_{first}") if c6.has("ES5", f"adapter_{first}") else c6.s.lora_init())
    return m


def es6(c6, d):
    """Oracle labels, test: bisection on wrong greedy traces of contested test problems."""
    ok, deadline = c6.budget("ES6")
    if not ok or not c6.has("ES1", "a1"):
        return {"ES6_skipped": SKIP if not ok else "skipped: no ES1 artifacts"}
    greedy, units, m = _oracle_stage(c6, d, "test", c6.v["test_cap"], "es6", deadline)
    correct = sorted([t for t in greedy if t["ok"] and t["bounds"]], key=O.sid)
    e5 = sorted(random.Random(5).sample(correct, min(E5_N, len(correct))), key=O.sid)
    c6.s.save(d, "units", units)
    c6.s.save(d, "e5", e5)
    m.update({"es6_units": len(units), "es6_correct_greedy": len(correct), "es6_e5_traces": len(e5)})
    m.update(oracle_summary(units, "es6_oracle"))
    return m


def _arm_rows(units, kind, inject_ids):
    return [(u["trace"], it["cut"], inject_ids if it["inject"] else []) for u, it in zip(units, O.arm_items(units, kind))]


def _stats(rows, groups, suffix=""):
    """Recovery per arm, deltas with cluster-bootstrap CIs (10,000 resamples by original_id), interaction. suffix
    names the adapter's arm columns (E3_rkl ...); E0/E1/E2 are shared."""
    out = {"n": len(rows), "problems": len(set(groups))}
    col = {a: a if a in BASE_ARMS else f"{a}{suffix}" for a in ARMS}
    for a in ARMS:
        out[f"rec_{a}"] = O.recovery(rows, col[a])
        out[f"tokens_{a}"] = statistics.mean(r[f"tok_{col[a]}"] for r in rows) if rows else float("nan")
        out[f"trunc_{a}"] = statistics.mean(r[f"trunc_{col[a]}"] for r in rows) if rows else float("nan")
    if not rows:
        return out
    E0, E1, E2, E3, E6 = col["E0"], col["E1"], col["E2"], col["E3"], col["E6"]
    stats = (("delta_primary", lambda rs: O.delta(rs, E3, E1)), ("e2_minus_e1", lambda rs: O.delta(rs, E2, E1)),
             ("e3_minus_e2", lambda rs: O.delta(rs, E3, E2)),
             ("interaction", lambda rs: O.delta(rs, E3, E1) - O.delta(rs, E6, E0)))
    for name, fn in stats:
        point, ci = O.cluster_bootstrap(rows, fn, groups)
        out[name], out[f"{name}_ci"] = point, ci
    return out


def es7(c6, d):
    """Arms on the test units with identical batches and chunk seeds (E0-E2 once, E3/E4/E6 per adapter); E5 per
    adapter on correct test traces; identity check with the untrained adapter."""
    from eval import continue_from
    from train_repair import set_lora
    ok, deadline = c6.budget("ES7")
    if not ok or not c6.has("ES6", "units"):
        return {"ES7_skipped": SKIP if not ok else "skipped: no ES6 units"}
    units = sorted(c6.load("ES6", "units"), key=lambda u: O.sid(u["trace"]))
    e5 = c6.load("ES6", "e5")
    model = c6.s.lora()
    kw = dict(bs=c6.s.gen_bs, max_new=MAX_NEW, eos=c6.s.eos, pad=c6.s.pad)
    set_lora(model, c6.s.lora_init())
    idn = units[:IDENT_N]
    seeds = O.chunk_seeds(c6.run_id, "ident", len(idn), c6.s.gen_bs)
    g1, _ = continue_from(model, _arm_rows(idn, "E1", []), seeds=seeds, adapter_on=False, **kw)
    g3, _ = continue_from(model, _arm_rows(idn, "E3", []), seeds=seeds, adapter_on=True, **kw)
    m = {"identity_n": len(idn), "identity_equal": sum(a == b for a, b in zip(g1, g3)),
         "identity_max_token_diff": max([sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
                                         for a, b in zip(g1, g3)], default=0)}
    adapters = [k for k in c6.v["adapters"] if c6.has("ES5", f"adapter_{k}")]
    m["es7_adapters"] = adapters
    m["es7_adapter_trained"] = bool(adapters)
    seeds = O.chunk_seeds(c6.run_id, "arms", len(units), c6.s.gen_bs)
    e5_rows = [(t, O.prefix_len(t["bounds"], O.false_cut(t, c6.run_id)), []) for t in e5]
    e5_seeds = O.chunk_seeds(c6.run_id, "e5", len(e5_rows), c6.s.gen_bs)
    # (column name, arm kind, adapter or None): E0-E2 adapter-free, then every adapter's arms + E5, then plain E5
    plan = [(a, a, None) for a in BASE_ARMS]
    for ad in adapters:
        plan += [(f"{a}_{ad}", a, ad) for a in ADAPTER_ARMS] + [(f"E5_{ad}", "E5", ad)]
    plan.append(("E5_plain", "E5", None))
    t0, n_tok, out, harm, cur = time.time(), 0, {}, {}, None
    for colname, arm, ad in plan:
        if time.time() > deadline:
            m["es7_arms_cut"] = f"deadline before {colname}"
            break
        if ad != cur:
            set_lora(model, c6.load("ES5", f"adapter_{ad}") if ad else c6.s.lora_init())
            cur = ad
        on = ad is not None
        if arm == "E5":
            toks, trunc = continue_from(model, e5_rows, seeds=e5_seeds, adapter_on=on, deadline=deadline, **kw)
            if len(toks) < len(e5_rows):
                m["es7_arms_cut"] = f"deadline inside {colname}"
                break
            harm[colname] = [not c6.unit_ok(t, cut, tk, tr) for (t, cut, _), tk, tr in zip(e5_rows, toks, trunc)]
        else:
            rows = _arm_rows(units, arm, c6.inject_ids)
            toks, trunc = continue_from(model, rows, seeds=seeds, adapter_on=on, deadline=deadline, **kw)
            if len(toks) < len(rows):
                m["es7_arms_cut"] = f"deadline inside {colname} ({len(toks)}/{len(rows)} rows)"
                break
            out[colname] = [(c6.unit_ok(t, cut, tk, tr), len(tk), tr) for (t, cut, _), tk, tr in zip(rows, toks, trunc)]
        n_tok += sum(len(tk) for tk in toks)
    m["es7_gen_tok_per_s"] = n_tok / max(time.time() - t0, 1e-9)
    rows = []
    for i, u in enumerate(units):
        t = u["trace"]
        r = {"sid": O.sid(t), "pid": t["pid"], "set": t["set"], "group": c6.c4.group[t["pid"]], "b": u["b"],
             "B": len(t["bounds"])}
        for colname in out:
            r[colname], r[f"tok_{colname}"], r[f"trunc_{colname}"] = out[colname][i]
        rows.append(r)
    c6.s.save(d, "arms", {"rows": rows, "harm": harm, "arms_done": list(out)})
    m.update({"es7_arms_done": list(out), "es7_units": len(rows), "es7_restart_units": sum(r["b"] == 0 for r in rows),
              "es7_pointed_units": sum(r["b"] >= 1 for r in rows), "es7_e5_n": len(e5_rows),
              "es7_clamped": {a: sum(it["clamped"] for it in O.arm_items(units, a)) for a in ("E4a", "E4b")},
              "es7_plain_harm": statistics.mean(harm["E5_plain"]) if harm.get("E5_plain") else float("nan")})
    regimes = {"all": lambda r: True, "pointed": lambda r: r["b"] >= 1, "restart": lambda r: r["b"] == 0}
    for ad in adapters:
        suffix = f"_{ad}"
        if not all(f"{a}{suffix}" in out for a in ADAPTER_ARMS) or not all(a in out for a in BASE_ARMS):
            continue
        m[f"es7_e5_harm_{ad}"] = statistics.mean(harm[f"E5_{ad}"]) if harm.get(f"E5_{ad}") else float("nan")
        for reg, keep in regimes.items():
            for s, rs in per_set([{"trace": {"set": r["set"]}, "row": r} for r in rows if keep(r)]).items():
                rr = [x["row"] for x in rs]
                m[f"es7_{ad}_{reg}_{s}"] = _stats(rr, [r["group"] for r in rr], suffix)
    if adapters:                                   # legacy keys mirror the first adapter, all units
        ad = adapters[0]
        m["es7_e5_harm"] = m.get(f"es7_e5_harm_{ad}", float("nan"))
        for s in ("p1", "p2", "main", "rc3", "pooled"):
            if f"es7_{ad}_all_{s}" in m:
                m[f"es7_{s}"] = m[f"es7_{ad}_all_{s}"]
    return m


def es8(c6, d):
    """Report + FULL-E projection (rewritten again after E-S3 by run_phase)."""
    import report
    report.option_e(c6.run_dir, c6.metrics, c6.cfg)
    return {}


def es3(c6, d):
    """Bisection validation: full scans on 20 bisected train traces; agreement within +-1."""
    ok, deadline = c6.budget("ES3")
    if not ok or not c6.has("ES2", "units"):
        return {"ES3_skipped": SKIP if not ok else "skipped: no ES2 units"}
    units = [u for u in c6.load("ES2", "units") if u["src"] == "bisect"]
    sel = random.Random(3).sample(units, min(VALID_N, len(units)))
    mk = c6.mark()
    full = O.run_full_scan(c6.mc_gen(deadline), c6.store, [u["trace"] for u in sel], deadline, k=c6.v["k"])
    m = c6.since(mk, "es3_mc")
    cmp = [(u["b"], O.oracle_cut(full[O.sid(u["trace"])], c6.v["thresh"])) for u in sel if O.sid(u["trace"]) in full]
    c6.s.save(d, "validation", {"pairs": cmp, "full": full})
    mono = [all(x >= y for x, y in zip(v, v[1:])) for v in full.values()]
    m.update({"es3_traces": len(cmp), "es3_of": len(sel),
              "es3_agree": statistics.mean(O.agree(a, b) for a, b in cmp) if cmp else float("nan"),
              "es3_exact": statistics.mean(a == b for a, b in cmp) if cmp else float("nan"),
              "es3_monotone_share": statistics.mean(mono) if mono else float("nan"),
              "es3_full_probes_per_trace": statistics.mean(len(v) for v in full.values()) if full else float("nan")})
    return m


STAGES = [("ES1", es1), ("ES2", es2), ("ES4", es4), ("ES5", es5), ("ES6", es6), ("ES7", es7), ("ES8", es8), ("ES3", es3)]
STAGES7 = [("ES0", es0)] + STAGES
