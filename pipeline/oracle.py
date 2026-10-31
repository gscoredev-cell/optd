"""Oracle pointer (OPTION_E_SPEC.md §2) and the Option E statistics (§5). Pure Python / numpy, no torch at import.

MC_b = share of k sampled completions from prefix(b) that reach the correct answer (b = 0: the prompt; b >= 1: the
generated tokens up to and including boundary b's newline token). Oracle cut b* = the last recoverable point:
0 if MC_0 < 0.5 (restart); B if MC_B >= 0.5 (recoverable to the end); else the largest b with MC_b >= 0.5 and
MC_{b+1} < 0.5. Bisection assumes MC is monotone non-increasing along a wrong trace and finds b* with ~2 + log2(B)
probes; `full_scan` probes every point. Probes are issued level-synchronously over many traces (one batch of
completions per round) and every completion goes through the generation cache, so a repeated probe is free.

An MC store (`/workspace/gencache/oracle_<model>.jsonl`) keeps finished MC vectors per (trace, point): SMOKE-4 A3 saved
only its MC results (the completions predate the cache), and E-S3's full scans must not pay for the bisection probes
twice.
"""
import hashlib
import json
import pathlib
import time

import numpy as np

from core import seed_for

MC_K, THRESH = 3, 0.5
NS = "optE"                       # seed namespace shared by every Option E run, so cache rows are reused across pods


def prefix_len(bounds, b):
    """Generated tokens kept for a cut at boundary b (0 = none)."""
    return 0 if b == 0 else bounds[b - 1] + 1


def oracle_cut(mc, thresh=THRESH):
    """b* from a full MC vector [MC_0, ..., MC_B] (§2); None when the vector is empty."""
    if not mc:
        return None
    B = len(mc) - 1
    if mc[0] < thresh:
        return 0
    if mc[B] >= thresh:
        return B
    return max(b for b in range(B) if mc[b] >= thresh and mc[b + 1] < thresh)


def bisect_step(probed, B, thresh=THRESH):
    """Next point to probe for one trace given {point: MC}, or ('done', b*). Order: 0, then B, then bisection between
    the highest recoverable point (lo) and the lowest unrecoverable one (hi)."""
    if 0 not in probed:
        return 0
    if probed[0] < thresh:
        return "done", 0
    if B not in probed:
        return B
    if probed[B] >= thresh:
        return "done", B
    lo = max(b for b, v in probed.items() if v >= thresh)
    hi = min(b for b, v in probed.items() if v < thresh and b > lo)
    if hi - lo <= 1:
        return "done", lo
    return (lo + hi) // 2


def bisect(mc_fn, B, thresh=THRESH):
    """Sequential bisection for one trace (tests and reference): mc_fn(point) -> MC. Returns (b*, probes)."""
    probed = {}
    while True:
        nxt = bisect_step(probed, B, thresh)
        if isinstance(nxt, tuple):
            return nxt[1], dict(probed)
        probed[nxt] = mc_fn(nxt)


def agree(b_bisect, b_full, tol=1):
    return abs(b_bisect - b_full) <= tol


def sid(t):
    return f"{t['pid']}#{t.get('k', 'g')}"


def _tsha(t):
    return hashlib.sha256(",".join(map(str, t["prompt"] + t["gen"])).encode()).hexdigest()


class MCStore:
    """Finished MC values per (trace sha, point) on the volume; one JSON line per value, later lines win."""

    def __init__(self, model_id, root=pathlib.Path("/workspace/gencache")):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"oracle_{model_id.replace('/', '__')}.jsonl"
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.rows[(e["sha"], e["b"])] = e["mc"]

    def get(self, t, b):
        return self.rows.get((_tsha(t), b))

    def put(self, items):
        """items: (trace, point, mc)."""
        lines = []
        for t, b, mc in items:
            self.rows[(_tsha(t), b)] = mc
            lines.append(json.dumps({"sha": _tsha(t), "b": b, "mc": mc}, separators=(",", ":")))
        if lines:
            with self.path.open("a") as f:
                f.write("\n".join(lines) + "\n")

    def import_vectors(self, pairs):
        """(trace, [MC_0..MC_B]) pairs from an earlier full scan (SMOKE-4 A3). Returns how many points were new."""
        new = [(t, b, v) for t, mc in pairs for b, v in enumerate(mc) if self.get(t, b) is None]
        self.put(new)
        return len(new)


def mc_rows(traces, points, k=MC_K):
    """Completion rows for the requested (trace, point) probes: (trace, point, prefix ids, row seed, budget)."""
    rows = []
    for t, b in zip(traces, points):
        cut = prefix_len(t["bounds"], b)
        for r in range(k):
            rows.append((t, b, t["prompt"] + t["gen"][:cut], seed_for(NS, "mc", f"{sid(t)}@{b}#{r}"), cut))
    return rows


def probe_points(gen, store, traces, points, k=MC_K):
    """MC at the given point of each trace through the store and the generation cache. gen(prefixes, seeds, budgets)
    -> (toks, trunc) with None for rows not generated; score(t, cut, toks, trunc) -> ok. Returns {sid: mc} for the
    traces whose k completions all finished."""
    gen_fn, score = gen
    out, todo = {}, []
    for t, b in zip(traces, points):
        v = store.get(t, b)
        if v is None:
            todo.append((t, b))
        else:
            out[sid(t)] = v
    if todo:
        rows = mc_rows([t for t, _ in todo], [b for _, b in todo], k)
        toks, trunc = gen_fn([r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows])
        succ = {}
        for (t, b, _, _, cut), tk, tr in zip(rows, toks, trunc):
            succ.setdefault(sid(t), []).append(None if tk is None else score(t, cut, tk, tr))
        fresh = []
        for t, b in todo:
            s = succ[sid(t)]
            if len(s) == k and all(x is not None for x in s):
                out[sid(t)] = sum(s) / k
                fresh.append((t, b, out[sid(t)]))
        store.put(fresh)
    return out


def run_bisection(gen, store, traces, deadline=None, thresh=THRESH, k=MC_K):
    """Level-synchronous bisection over traces. Returns ({sid: {"b": b*, "probed": {point: mc}}}, rounds); traces
    whose probes did not finish before the deadline are left out."""
    state = {sid(t): {} for t in traces}
    by = {sid(t): t for t in traces}
    done, rounds = {}, 0
    while True:
        want = {}
        for s, probed in state.items():
            if s in done:
                continue
            nxt = bisect_step(probed, len(by[s]["bounds"]), thresh)
            if isinstance(nxt, tuple):
                done[s] = {"b": nxt[1], "probed": dict(probed)}
            else:
                want[s] = nxt
        if not want or (deadline and time.time() > deadline):
            break
        rounds += 1
        got = probe_points(gen, store, [by[s] for s in want], [want[s] for s in want], k)
        for s, v in got.items():
            state[s][want[s]] = v
        if len(got) < len(want):   # out of time inside the batch: finish what has its value, drop the rest
            for s in list(want):
                if s not in got:
                    state.pop(s)
    return done, rounds


def run_full_scan(gen, store, traces, deadline=None, k=MC_K):
    """MC at every point of each trace in one level-synchronous sweep per point index. {sid: [MC_0..MC_B]}."""
    by = {sid(t): t for t in traces}
    vec = {s: [] for s in by}
    maxB = max((len(t["bounds"]) for t in traces), default=-1)
    for b in range(maxB + 1):
        if deadline and time.time() > deadline:
            break
        ts = [by[s] for s in vec if len(by[s]["bounds"]) >= b]
        got = probe_points(gen, store, ts, [b] * len(ts), k)
        for s in list(vec):
            if len(by[s]["bounds"]) >= b:
                if s in got:
                    vec[s].append(got[s])
                else:
                    vec.pop(s)
    return {s: v for s, v in vec.items() if len(v) == len(by[s]["bounds"]) + 1}


# ---- statistics (§5) ------------------------------------------------------------------------------------------------

def cluster_bootstrap(rows, stat, clusters, n=10000, seed=0):
    """Cluster bootstrap (resample clusters with replacement) of stat(rows_subset) -> float. rows: list of dicts;
    clusters: cluster id per row. Returns (point, [lo, hi]) with a 95% percentile interval."""
    ids = sorted(set(clusters))
    members = {c: [i for i, g in enumerate(clusters) if g == c] for c in ids}
    rng = np.random.default_rng(seed)
    point = stat(rows)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(ids), len(ids), replace=True)
        vals.append(stat([rows[i] for c in pick for i in members[ids[c]]]))
    vals = np.array(vals, dtype=float)
    return float(point), [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def recovery(rows, arm):
    return float(np.mean([r[arm] for r in rows])) if rows else float("nan")


def delta(rows, a, b):
    return recovery(rows, a) - recovery(rows, b)


def interaction(rows):
    """I = [rec(E3) - rec(E1)] - [rec(E6) - rec(E0)] (§5.4)."""
    return delta(rows, "E3", "E1") - delta(rows, "E6", "E0")


def arm_items(units, kind):
    """Per-arm start points over the same unit list (sorted by sid), §3. kind -> (prefix kept, adapter on, inject).
    Every arm gets exactly one row per unit in the same order, so batches and chunk seeds pair across arms."""
    out = []
    for u in units:
        t, b, B = u["trace"], u["b"], len(u["trace"]["bounds"])
        start = {"E0": 0, "E1": b, "E2": b, "E3": b, "E4a": max(b - 1, 0), "E4b": min(b + 1, B), "E6": 0}[kind]
        out.append({"sid": sid(t), "cut": prefix_len(t["bounds"], start), "adapter": kind in ("E3", "E4a", "E4b", "E6"),
                    "inject": kind == "E2", "clamped": kind in ("E4a", "E4b") and start == b})
    return out


def chunk_seeds(run_id, tag, n_rows, bs):
    """One sampling seed per batch chunk, identical for every arm (same tag)."""
    return [seed_for(run_id, tag, c) for c in range((n_rows + bs - 1) // bs)]


def false_cut(t, run_id):
    """E5: random cut uniform over 1..B, seeded per trace."""
    import random
    B = len(t["bounds"])
    return random.Random(seed_for(run_id, "e5cut", sid(t))).randint(1, B) if B else 0
