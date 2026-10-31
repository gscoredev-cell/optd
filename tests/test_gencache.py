"""gencache: hits, misses, the cap rule and reload, with a fake decode (no torch)."""
from gencache import GREEDY, GenCache, cached_decode

EOS = 99


def _fake(lengths):
    """decode stand-in: row i of a call emits lengths[first token of the row] tokens, then EOS (if within max_new)."""
    calls = []

    def decode(model, seqs, max_new, eos, pad, sample=False, seed=0, **kw):
        calls.append((len(seqs), max_new, sample, seed))
        rows = [([7] * lengths[s[0]] + [EOS])[:max_new] for s in seqs]
        return rows, None, None
    return decode, calls


def test_miss_then_hit_and_reload(tmp_path):
    c = GenCache("org/M", root=tmp_path)
    dec, calls = _fake({1: 3, 2: 5, 3: 2})
    seqs, seeds = [[1], [2], [3]], [11, 12, 13]
    toks, trunc = cached_decode(c, None, seqs, seeds, 2, 800, {EOS}, 0, sample=True, decode_fn=dec)
    assert toks[0] == [7, 7, 7, EOS] and trunc == [False] * 3 and c.stats() == {"hits": 0, "misses": 3}
    assert [n for n, *_ in calls] == [2, 1] and calls[0][3] != calls[1][3]
    c2 = GenCache("org/M", root=tmp_path)                       # reload from the jsonl
    toks2, _ = cached_decode(c2, None, seqs, seeds, 2, 800, {EOS}, 0, sample=True, decode_fn=dec)
    assert toks2 == toks and c2.stats() == {"hits": 3, "misses": 0} and len(calls) == 2
    _, _ = cached_decode(c2, None, seqs, [21, 22, 23], 2, 800, {EOS}, 0, sample=True, decode_fn=dec)
    assert c2.misses == 3                                       # another seed is another draw
    _, _ = cached_decode(c2, None, seqs, [0, 0, 0], 2, 800, {EOS}, 0, sample=False, decode_fn=dec)
    assert c2.misses == 6                                       # greedy is keyed apart from sampling


def test_cap_rule(tmp_path):
    c = GenCache("m", root=tmp_path)
    c.put_many([([1], *GREEDY, 0, 640, [7] * 640, True),        # truncated at 640
                ([2], *GREEDY, 0, 640, [7, 7, EOS], False),     # ended before 640
                ([3], *GREEDY, 0, 800, [7] * 10 + [EOS], False)])
    assert c.get([1], *GREEDY, 0, 800) is None                  # truncated under a lower cap -> regenerate
    assert c.get([1], *GREEDY, 0, 640) == ([7] * 640, True)
    assert c.get([2], *GREEDY, 0, 800) == ([7, 7, EOS], False)  # finished: valid under a raised cap
    assert c.get([3], *GREEDY, 0, 5) == ([7] * 5, True)         # trimmed to a lower cap -> truncated there
    dec, calls = _fake({1: 700, 2: 2, 3: 10})
    toks, trunc = cached_decode(c, None, [[1], [2], [3]], [0, 0, 0], 8, 800, {EOS}, 0, decode_fn=dec)
    assert c.stats() == {"hits": 2, "misses": 1} and len(toks[0]) == 701 and trunc[0] is False
    assert c.get([1], *GREEDY, 0, 800) == (toks[0], False)      # the regenerated row replaced the old one


def test_budgets_and_deadline(tmp_path):
    c = GenCache("m", root=tmp_path)
    dec, _ = _fake({1: 50, 2: 50})
    toks, trunc = cached_decode(c, None, [[1], [2]], [1, 2], 1, 800, {EOS}, 0, budgets=[10, 0], sample=True,
                                decode_fn=dec)
    assert [len(t) for t in toks] == [10, 1] and trunc == [True, True]
    toks, trunc = cached_decode(c, None, [[1], [9]], [5, 6], 1, 800, {EOS}, 0, deadline=1, sample=True,
                                decode_fn=dec)
    assert toks == [None, None] and c.misses == 2               # past the deadline nothing new starts
