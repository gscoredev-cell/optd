"""S8 arms (§9.3, §9.5). A0 = the S1 greedy trace. A2/A3: flagged traces are cut at the first flag (offline-
equivalent rollback) and the continuation is resampled, A2 without and A3 with the adapter, in identical batches with
identical seeds. A6: adapter on from the first generated token, greedy, no probe."""
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
