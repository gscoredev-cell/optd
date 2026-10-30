"""S5: k continuations per flagged prefix, adapter on, T = 0.8 / top-p 0.95 (§9.4)."""
from core import seed_for
from generate import decode_batches


def continuations(student, items, k, bs, max_new, eos, pad, run_id):
    """items: dicts {pid, prefix (prompt + kept generated ids), cut (kept generated tokens)}, sorted by pid.
    Returns one dict per continuation {pid, j, prefix, cut, cont, truncated}."""
    rows = [(it, j) for it in items for j in range(k)]
    toks, trunc = decode_batches(
        student, [it["prefix"] for it, _ in rows], bs, max_new, eos, pad,
        seed_fn=lambda c: seed_for(run_id, "cont", rows[c * bs][0]["pid"]),
        budgets=[max_new - it["cut"] for it, _ in rows], sample=True, adapter_on=True)
    return [{"pid": it["pid"], "j": j, "prefix": it["prefix"], "cut": it["cut"], "cont": t, "truncated": tr}
            for (it, j), t, tr in zip(rows, toks, trunc)]
