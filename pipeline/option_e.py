"""SMOKE-6 (OPTION_E_SPEC.md §6): trained repair under an oracle pointer. Stages in the deadline order
E-S1, E-S2, E-S4, E-S5, E-S6, E-S7, E-S8 (report), then E-S3 (bisection validation, optional; the report is rewritten
once more after it). Every stage has a time budget (L40 estimates x 1.3) and reserves the budgets of the required
stages after it; a stage that cannot start in time is skipped and says so.

Data: SMOKE-4 A1 traces (400 train + 400 test problems x 4 samples, Qwen3-1.7B, T 0.8, cap 800) define the contested
stratum (1..3 of 4 correct) and are imported into the generation cache; A3's 48 full MC scans are imported as MC
vectors (the completions were not stored). First attempts are greedy (through the cache). Train units = wrong greedy
traces of contested train problems (bisection) + the A3-scanned sampled wrong traces (own scans, §4) + sampled wrong
traces whose first b* boundaries share >= 70% of their tokens with the greedy trace (reuse its cut). Test units = wrong
greedy traces of contested test problems. Adapter-on generation never goes through the cache (the adapter changes it).
torch is imported lazily so the pure parts (arm construction, statistics) are testable without it."""
import collections
import pathlib
import random
import statistics
import time

import oracle as O
from core import primary_ok, seed_for
from gencache import GREEDY, GenCache, cached_decode

MAX_NEW = 800
INJECT = "\nWait — let me re-check that step.\n"
TRAIN_CAP, TEST_CAP, DEV_N, E5_N, VALID_N, IDENT_N = 200, 120, 60, 60, 20, 20
CONT_K, SLOPPY, ROUNDS, OVERLAP = 4, 0.25, 2, 0.70
# minutes: spec L40 estimates x 1.3 (E-S8 = report), and the least time a stage may start with
BUDGET_MIN = {"ES1": 4, "ES2": 33, "ES4": 11, "ES5": 18, "ES6": 20, "ES7": 16, "ES8": 3, "ES3": 8}
START_MIN = {"ES1": 2, "ES2": 8, "ES4": 4, "ES5": 5, "ES6": 6, "ES7": 6, "ES8": 1, "ES3": 3}
ORDER = ["ES1", "ES2", "ES4", "ES5", "ES6", "ES7", "ES8"]      # required stages, in run order; ES3 after them
SKIP = "skipped: not enough pod time left"
ARMS = ("E0", "E1", "E2", "E3", "E4a", "E4b", "E6")


class Ctx6:
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        from smoke4 import Ctx4
        self.cfg, self.run_id, self.run_dir = cfg, run_id, run_dir
        self.c4 = Ctx4(cfg, tiers, tier, run_id, run_dir)      # probs (800), split, group, gold; gen_bs x 4
        self.s = self.c4.s
        self.s.gen_bs = min(self.s.gen_bs, 128)              # 1.7B KV at cap 800: 128 rows ~ 23 GB
        self.src = pathlib.Path("/workspace/runs") / cfg["smoke4_source_run"]
        self.cache = GenCache(cfg["student"])
        self.store = O.MCStore(cfg["student"])
        self.inject_ids = self.s.tok.encode(INJECT, add_special_tokens=False)

    # time
    def budget(self, name):
        """(start allowed, stage deadline); the deadline leaves every later required stage its nominal budget."""
        from smoke3 import pipeline_deadline
        later = ORDER[ORDER.index(name) + 1:] if name in ORDER else ["ES8"]
        end = pipeline_deadline() - sum(BUDGET_MIN[r] for r in later) * 60
        return time.time() + START_MIN[name] * 60 <= end, min(time.time() + BUDGET_MIN[name] * 60, end)

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


def es1(c6, d):
    """Import SMOKE-4 A1 traces (cache) and A3 scans (MC store); select contested problems; counts."""
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
    a3_new = c6.store.import_vectors(a3)
    bp = by_problem(traces)
    cont = {p: ts for p, ts in bp.items() if contested(ts)}
    sel = [dict(t, split=c6.c4.split[t["pid"]]) for p in sorted(cont) for t in cont[p]]
    c6.s.save(d, "a1", sel)
    c6.s.save(d, "a3", [{"trace": dict(t, split=c6.c4.split[t["pid"]]), "mc": mc, "b": O.oracle_cut(mc)}
                        for t, mc in a3])
    m = {"es1_a1_traces": len(traces), "es1_a1_imported": len(new), "es1_a3_scans": len(a3), "es1_a3_points_new": a3_new,
         "es1_a3_bad_length": len(orc) - len(a3)}
    for sp in ("train", "test"):
        ps = [p for p in cont if c6.c4.split[p] == sp]
        m[f"es1_contested_{sp}"] = {**{s: sum(p.split(":")[0] == s for p in ps) for s in ("p1", "p2", "main")},
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
    done, rounds = O.run_bisection(c6.mc_gen(deadline), c6.store, wrong, deadline)
    m.update(c6.since(mk, f"{tag}_mc"))
    units = [{"trace": t, "b": done[O.sid(t)]["b"], "probed": done[O.sid(t)]["probed"], "src": "bisect"}
             for t in wrong if O.sid(t) in done]
    m.update({f"{tag}_bisect_traces": len(units), f"{tag}_bisect_of": len(wrong), f"{tag}_bisect_rounds": rounds,
              f"{tag}_probes_per_trace": statistics.mean(len(u["probed"]) for u in units) if units else float("nan")})
    return greedy, units, m


def es2(c6, d):
    """Oracle labels, train: bisection on wrong greedy traces of contested train problems; A3 units reused."""
    ok, deadline = c6.budget("ES2")
    if not ok or not c6.has("ES1", "a1"):
        return {"ES2_skipped": SKIP if not ok else "skipped: no ES1 artifacts"}
    greedy, units, m = _oracle_stage(c6, d, "train", TRAIN_CAP, "es2", deadline)
    by_g = {t["pid"]: t for t in greedy}
    a3 = [dict(u, src="a3") for u in c6.load("ES1", "a3") if u["b"] is not None]
    seen = {O.sid(u["trace"]) for u in units}
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
    import torch
    from generate import load_model, token_logprobs
    from smoke import free
    rows = [(u, j) for u in units for j in range(CONT_K)]
    seqs = [u["trace"]["prompt"] + u["trace"]["gen"][:O.prefix_len(u["trace"]["bounds"], u["b"])] for u, _ in rows]
    cuts = [len(s) - len(u["trace"]["prompt"]) for s, (u, _) in zip(seqs, rows)]
    from generate import decode_batches
    t0 = time.time()
    toks, trunc = decode_batches(model, seqs, c6.s.gen_bs, MAX_NEW, c6.s.eos, c6.s.pad,
                                 seed_fn=lambda c: seed_for(c6.run_id, f"cont{tag}", c),
                                 budgets=[MAX_NEW - c for c in cuts], sample=True, adapter_on=True, deadline=deadline)
    gen_sec = time.time() - t0
    rows, seqs, cuts = rows[:len(toks)], seqs[:len(toks)], cuts[:len(toks)]
    ex = [{"prefix": s, "cont": tk, "cut": c, "sid": O.sid(u["trace"]), "sloppy": u.get("sloppy", False),
           "ok": c6.unit_ok(u["trace"], c, tk, tr), "truncated": tr}
          for (u, _), s, tk, c, tr in zip(rows, seqs, toks, cuts, trunc) if tk]
    ex = [e for e in ex if e["cont"]]
    lp_s = token_logprobs(model, [e["prefix"] for e in ex], [e["cont"] for e in ex], c6.s.pre_bs, adapter_on=True)
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
              f"{tag}_gen_tok_per_s": sum(len(e["cont"]) for e in ex) / max(gen_sec, 1e-9),
              f"{tag}_prefill_tok_per_s": sum(len(e["prefix"]) + len(e["cont"]) for e in ex) / max(prefill_sec, 1e-9),
              f"{tag}_teacher_also_wrong": statistics.mean(twrong) if twrong else float("nan"),
              f"{tag}_teacher_scored": len(twrong), f"{tag}_mean_adv": statistics.mean(adv) if adv else float("nan")})
    return ex


def _split_dev(units):
    """60 dev traces (one unit each, seeded) never used for training; every unit of a dev trace's problem is kept out
    of training (dev is split by problem, like test)."""
    traces = sorted({O.sid(u["trace"]): u for u in units}.items())
    dev_sids = set(s for s, _ in random.Random(60).sample(traces, min(DEV_N, len(traces))))
    dev = [u for s, u in traces if s in dev_sids]
    dev_pids = {u["trace"]["pid"] for u in dev}
    return dev, [u for u in units if u["trace"]["pid"] not in dev_pids]


def es4(c6, d):
    """Round-1 continuations (+25% sloppy cuts) from the oracle prefixes with the untrained adapter; teacher scoring."""
    ok, deadline = c6.budget("ES4")
    if not ok or not c6.has("ES2", "units"):
        return {"ES4_skipped": SKIP if not ok else "skipped: no ES2 units"}
    dev, train_units = _split_dev(c6.load("ES2", "units"))
    from train_repair import sloppy_mix
    train_units = sloppy_mix(train_units, SLOPPY, seed_for(c6.run_id, "sloppy", 0))
    c6.s.save(d, "dev", dev)
    c6.s.save(d, "train_units", train_units)
    model = c6.s.lora()
    m = {"es4_dev_traces": len(dev), "es4_dev_b_ge1": sum(u["b"] >= 1 for u in dev)}
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


def es5(c6, d):
    """Train rkl with early stopping on the dev set, 2 rounds (round 2 resamples with the current adapter); the
    kill-at-15/resume check runs in round 1."""
    import math
    import shutil
    import torch
    from core import kill_plan
    from train_repair import lora_state, set_lora, train
    ok, deadline = c6.budget("ES5")
    if not ok or not c6.has("ES4", "examples"):
        return {"ES5_skipped": SKIP if not ok else "skipped: no ES4 examples"}
    model, dev, units = c6.s.lora(), c6.load("ES4", "dev"), c6.load("ES4", "train_units")
    ex = c6.load("ES4", "examples")
    eval_fn = _dev_eval(c6, dev)
    set_lora(model, c6.s.lora_init())
    m = {"train_examples_r1": len(ex), "adapter_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    best, curves, t0 = None, [], time.time()
    for rnd in range(1, ROUNDS + 1):
        if rnd > 1:
            if time.time() + 240 > deadline:
                m[f"round{rnd}_skipped"] = "stage time budget"
                break
            set_lora(model, best["state"])
            ex = _round(c6, units, model, f"es5r{rnd}", deadline, m)
            c6.s.save(d, f"examples_r{rnd}", ex)
            m[f"train_examples_r{rnd}"] = len(ex)
            if time.time() + 120 > deadline:
                m[f"round{rnd}_skipped"] = "stage time budget (continuations sampled, training skipped)"
                break
        if not ex:
            m["train_blocker"] = "no training examples"
            break
        ck = d / f"ckpt_r{rnd}"
        shutil.rmtree(ck, ignore_errors=True)
        steps = math.ceil(len(ex) / 16)
        early = {}
        plan = kill_plan(steps) if rnd == 1 else None
        if plan:
            c, k = plan
            first, _ = train(model, ex, ck, c6.run_id, stop_at=k, ckpt_every=c, eval_fn=eval_fn, early=early)
            set_lora(model, c6.s.lora_init())             # the "killed" process's weights are gone; resume restores them
            losses, _ = train(model, ex, ck, c6.run_id, resume=True, ckpt_every=c, eval_fn=eval_fn, early=early)
            gap = [abs(first[s] - losses[s]) / max(1.0, abs(first[s])) for s in range(c + 1, k + 1) if s in losses]
            m.update({"train_ckpt_every": c, "train_kill_at": k, "resume_max_rel_diff": max(gap, default=float("nan")),
                      "resume_ok": bool(gap) and max(gap) <= 1e-3})
        else:
            losses, _ = train(model, ex, ck, c6.run_id, ckpt_every=10, eval_fn=eval_fn, early=early)
            if rnd == 1:
                m.update({"resume_ok": None, "train_blocker": f"only {steps} steps: kill/resume check needs >= 5"})
        curves.append({"round": rnd, "steps": steps, "curve": early.get("curve", []), "best_step": early.get("best_step"),
                       "best_dev": early.get("best_score"), "stopped": early.get("stopped"),
                       "loss_start": losses.get(1), "loss_end": losses.get(max(losses)) if losses else None})
        if early.get("best_state") is not None and (best is None or early["best_score"] > best["dev"]):
            best = {"dev": early["best_score"], "step": early.get("best_step"), "round": rnd,
                    "state": early["best_state"]}
        m["train_steps_total"] = m.get("train_steps_total", 0) + early.get("last_step", 0)
    sec = time.time() - t0
    if best is None:
        best = {"dev": None, "step": 0, "round": 0, "state": c6.s.lora_init()}
    torch.save(best["state"], d / "adapter.pt")
    set_lora(model, best["state"])
    m.update({"train_rounds": curves, "train_rounds_done": len(curves), "train_best_dev": best["dev"],
              "train_best_step": best["step"], "train_best_round": best["round"],
              "early_stop_fired": any(c["stopped"] for c in curves),
              "train_steps_per_s": m.get("train_steps_total", 0) / max(sec, 1e-9), "train_sec": sec})
    return m


def es6(c6, d):
    """Oracle labels, test: bisection on wrong greedy traces of contested test problems (<= 120)."""
    ok, deadline = c6.budget("ES6")
    if not ok or not c6.has("ES1", "a1"):
        return {"ES6_skipped": SKIP if not ok else "skipped: no ES1 artifacts"}
    greedy, units, m = _oracle_stage(c6, d, "test", TEST_CAP, "es6", deadline)
    correct = sorted([t for t in greedy if t["ok"] and t["bounds"]], key=O.sid)
    e5 = sorted(random.Random(5).sample(correct, min(E5_N, len(correct))), key=O.sid)
    c6.s.save(d, "units", units)
    c6.s.save(d, "e5", e5)
    m.update({"es6_units": len(units), "es6_correct_greedy": len(correct), "es6_e5_traces": len(e5)})
    m.update(oracle_summary(units, "es6_oracle"))
    return m


def _arm_rows(units, kind, inject_ids):
    return [(u["trace"], it["cut"], inject_ids if it["inject"] else []) for u, it in zip(units, O.arm_items(units, kind))]


def _stats(rows, groups):
    """Recovery per arm, deltas with cluster-bootstrap CIs (10,000 resamples by original_id), interaction."""
    out = {"n": len(rows), "problems": len(set(groups))}
    for a in ARMS:
        out[f"rec_{a}"] = O.recovery(rows, a)
        out[f"tokens_{a}"] = statistics.mean(r[f"tok_{a}"] for r in rows) if rows else float("nan")
        out[f"trunc_{a}"] = statistics.mean(r[f"trunc_{a}"] for r in rows) if rows else float("nan")
    if not rows:
        return out
    for name, fn in (("delta_primary", lambda rs: O.delta(rs, "E3", "E1")), ("e2_minus_e1", lambda rs: O.delta(rs, "E2", "E1")),
                     ("e3_minus_e2", lambda rs: O.delta(rs, "E3", "E2")), ("interaction", O.interaction)):
        point, ci = O.cluster_bootstrap(rows, fn, groups)
        out[name], out[f"{name}_ci"] = point, ci
    return out


def es7(c6, d):
    """Arms E0-E6 on the test units with identical batches and chunk seeds; E5 on correct test traces; identity check."""
    import torch
    from eval import continue_from
    from train_repair import set_lora
    ok, deadline = c6.budget("ES7")
    if not ok or not c6.has("ES6", "units"):
        return {"ES7_skipped": SKIP if not ok else "skipped: no ES6 units"}
    units = sorted(c6.load("ES6", "units"), key=lambda u: O.sid(u["trace"]))
    e5 = c6.load("ES6", "e5")
    model = c6.s.lora()
    kw = dict(bs=c6.s.gen_bs, max_new=MAX_NEW, eos=c6.s.eos, pad=c6.s.pad)
    # identity: untrained E3 == E1 in identical batches
    set_lora(model, c6.s.lora_init())
    idn = units[:IDENT_N]
    seeds = O.chunk_seeds(c6.run_id, "ident", len(idn), c6.s.gen_bs)
    g1, _ = continue_from(model, _arm_rows(idn, "E1", []), seeds=seeds, adapter_on=False, **kw)
    g3, _ = continue_from(model, _arm_rows(idn, "E3", []), seeds=seeds, adapter_on=True, **kw)
    m = {"identity_n": len(idn), "identity_equal": sum(a == b for a, b in zip(g1, g3)),
         "identity_max_token_diff": max([sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
                                         for a, b in zip(g1, g3)], default=0)}
    trained = c6.has("ES5", "adapter")
    set_lora(model, c6.load("ES5", "adapter") if trained else c6.s.lora_init())
    m["es7_adapter_trained"] = trained
    seeds = O.chunk_seeds(c6.run_id, "arms", len(units), c6.s.gen_bs)
    t0, n_tok, out = time.time(), 0, {}
    for arm in ARMS:
        if time.time() > deadline:
            m["es7_arms_cut"] = f"deadline before {arm}"
            break
        rows = _arm_rows(units, arm, c6.inject_ids)
        toks, trunc = continue_from(model, rows, seeds=seeds, adapter_on=arm in ("E3", "E4a", "E4b", "E6"),
                                    deadline=deadline, **kw)
        if len(toks) < len(rows):
            m["es7_arms_cut"] = f"deadline inside {arm} ({len(toks)}/{len(rows)} rows)"
            break
        out[arm] = [(c6.unit_ok(t, cut, tk, tr), len(tk), tr) for (t, cut, _), tk, tr in zip(rows, toks, trunc)]
        n_tok += sum(len(tk) for tk in toks)
    m["es7_gen_tok_per_s"] = n_tok / max(time.time() - t0, 1e-9)
    done = [a for a in ARMS if a in out]
    rows = []
    for i, u in enumerate(units):
        t = u["trace"]
        r = {"sid": O.sid(t), "pid": t["pid"], "set": t["set"], "group": c6.c4.group[t["pid"]], "b": u["b"],
             "B": len(t["bounds"])}
        for a in done:
            r[a], r[f"tok_{a}"], r[f"trunc_{a}"] = out[a][i]
        rows.append(r)
    clamped = {a: sum(it["clamped"] for it in O.arm_items(units, a)) for a in ("E4a", "E4b")}
    # E5: correct greedy traces, random cut 1..B seeded per trace; adapter on vs plain retry at the same cuts
    e5_rows = [(t, O.prefix_len(t["bounds"], O.false_cut(t, c6.run_id)), []) for t in e5]
    e5_seeds = O.chunk_seeds(c6.run_id, "e5", len(e5_rows), c6.s.gen_bs)
    harm = {}
    if time.time() < deadline and e5_rows:
        for name, on in (("E5", True), ("plain", False)):
            toks, trunc = continue_from(model, e5_rows, seeds=e5_seeds, adapter_on=on, deadline=deadline, **kw)
            if len(toks) < len(e5_rows):
                harm.pop("E5", None)
                m["es7_e5_cut"] = f"deadline inside E5 {name}"
                break
            harm[name] = [not c6.unit_ok(t, cut, tk, tr) for (t, cut, _), tk, tr in zip(e5_rows, toks, trunc)]
    c6.s.save(d, "arms", {"rows": rows, "harm": harm, "arms_done": done})
    m["es7_arms_done"] = done
    if all(a in done for a in ARMS):
        for s, rs in per_set([{"trace": {"set": r["set"]}, "row": r} for r in rows]).items():
            rr = [x["row"] for x in rs]
            m[f"es7_{s}"] = _stats(rr, [r["group"] for r in rr])
    m.update({"es7_clamped": clamped, "es7_units": len(rows), "es7_restart_units": sum(r["b"] == 0 for r in rows),
              "es7_e5_n": len(e5_rows),
              "es7_e5_harm": statistics.mean(harm["E5"]) if harm.get("E5") else float("nan"),
              "es7_plain_harm": statistics.mean(harm["plain"]) if harm.get("plain") else float("nan")})
    return m


def es8(c6, d):
    """Report + FULL-E projection (rewritten again after E-S3 by run_phase)."""
    import report
    m = {}
    report.option_e(c6.run_dir, c6.metrics, c6.cfg)
    return m


def es3(c6, d):
    """Bisection validation: full scans on 20 bisected train traces; agreement within +-1."""
    ok, deadline = c6.budget("ES3")
    if not ok or not c6.has("ES2", "units"):
        return {"ES3_skipped": SKIP if not ok else "skipped: no ES2 units"}
    units = [u for u in c6.load("ES2", "units") if u["src"] == "bisect"]
    sel = random.Random(3).sample(units, min(VALID_N, len(units)))
    mk = c6.mark()
    full = O.run_full_scan(c6.mc_gen(deadline), c6.store, [u["trace"] for u in sel], deadline)
    m = c6.since(mk, "es3_mc")
    cmp = [(u["b"], O.oracle_cut(full[O.sid(u["trace"])])) for u in sel if O.sid(u["trace"]) in full]
    c6.s.save(d, "validation", {"pairs": cmp, "full": full})
    mono = [all(x >= y for x, y in zip(v, v[1:])) for v in full.values()]
    m.update({"es3_traces": len(cmp), "es3_of": len(sel),
              "es3_agree": statistics.mean(O.agree(a, b) for a, b in cmp) if cmp else float("nan"),
              "es3_exact": statistics.mean(a == b for a, b in cmp) if cmp else float("nan"),
              "es3_monotone_share": statistics.mean(mono) if mono else float("nan"),
              "es3_full_probes_per_trace": statistics.mean(len(v) for v in full.values()) if full else float("nan")})
    return m


STAGES = [("ES1", es1), ("ES2", es2), ("ES4", es4), ("ES5", es5), ("ES6", es6), ("ES7", es7), ("ES8", es8), ("ES3", es3)]
