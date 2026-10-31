"""S8 arms (§9.3, §9.5) and the Option E arms (OPTION_E_SPEC §3). A0 = the S1 greedy trace. A2/A3: flagged traces are
cut at the first flag (offline-equivalent rollback) and the continuation is resampled, A2 without and A3 with the
adapter, in identical batches with identical seeds. A6: adapter on from the first generated token, greedy, no probe.
continue_from(): one continuation per explicit (prefix, cut) row with per-chunk seeds, the primitive the E arms share."""
from core import rollback_cut, seed_for
from generate import decode_batches


def retry(model, a0, flags, bs, max_new, eos, pad, run_id, adapter_on):
    """a0: traces sorted by pid with prompt, gen, bounds; flags: pid -> first flagged boundary (1-based) or None.
    Returns pid -> (full generated ids, truncated) for flagged pids."""
    items = [(t, rollback_cut(t["bounds"], flags[t["pid"]])) for t in a0 if flags.get(t["pid"])]
    if not items:
        return {}
    toks, trunc = decode_batches(
        model, [t["prompt"] + t["gen"][:cut] for t, cut in items], bs, max_new, eos, pad,
        seed_fn=lambda c: seed_for(run_id, "retry", items[c * bs][0]["pid"]),
        budgets=[max_new - cut for _, cut in items], sample=True, adapter_on=adapter_on)
    return {t["pid"]: (t["gen"][:cut] + tk, tr) for (t, cut), tk, tr in zip(items, toks, trunc)}


def plain(model, a0, bs, max_new, eos, pad, adapter_on):
    """Greedy from the prompt for every trace (A6 with adapter_on=True). pid -> (generated ids, truncated)."""
    toks, trunc = decode_batches(model, [t["prompt"] for t in a0], bs, max_new, eos, pad, adapter_on=adapter_on)
    return {t["pid"]: (tk, tr) for t, tk, tr in zip(a0, toks, trunc)}


def continue_from(model, rows, bs, max_new, eos, pad, seeds, adapter_on, deadline=None):
    """rows: (trace, kept generated tokens, extra ids appended after the kept prefix). One sampled continuation per
    row (T 0.8, top_p 0.95) in chunks of bs with seeds[c] per chunk, so two calls with the same rows and seeds pair
    exactly. Returns (new tokens incl. the extra ids, truncated) per row; rows past the deadline are missing."""
    seqs = [t["prompt"] + t["gen"][:cut] + extra for t, cut, extra in rows]
    toks, trunc = decode_batches(model, seqs, bs, max_new, eos, pad, seed_fn=lambda c: seeds[c],
                                 budgets=[max_new - cut - len(extra) for _, cut, extra in rows], sample=True,
                                 adapter_on=adapter_on, deadline=deadline)
    return [extra + tk for (_, _, extra), tk in zip(rows, toks)], trunc
