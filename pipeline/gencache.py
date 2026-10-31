"""Generation cache on the volume (SMOKE-5 on): /workspace/gencache/<model>.jsonl, one JSON line per generated row.

Key = (model id, sha256 of the input token ids, temperature, top_p, row seed); each entry also stores its max_new_tokens
cap, the new token ids and the truncated flag (never text). A lookup at cap c hits an entry made at cap >= c (trimmed to
c) or at a lower cap if that row ended before its cap. A row truncated under a lower cap is regenerated: the only
reason to regenerate (never to change scoring, labels or thresholds). Sampling seeds one RNG per batch
(generate.decode), so a row seed names a draw, it does not replay it bit for bit. Later lines win on reload. No torch.
"""
import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path("/workspace/gencache")
GREEDY = (0.0, 1.0)   # (temperature, top_p) recorded for greedy rows; their seed is 0


def _sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


class GenCache:
    def __init__(self, model_id, root=ROOT):
        self.model = model_id
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{model_id.replace('/', '__')}.jsonl"
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue                   # a torn last line from a killed pod
                self.rows[e["key"]] = e
        self.hits = self.misses = self.gen_tokens = 0

    def key(self, ids, temperature, top_p, seed):
        return f"{self.model}|{_sha(ids)}|T{temperature:g}|p{top_p:g}|s{seed}"

    def get(self, ids, temperature, top_p, seed, cap):
        """(new tokens, truncated) at this cap, or None (absent, or truncated under a lower cap)."""
        e = self.rows.get(self.key(ids, temperature, top_p, seed))
        if e is None or (e["cap"] < cap and e["trunc"]):
            return None
        return e["gen"][:cap], e["trunc"] or len(e["gen"]) > cap

    def put_many(self, rows):
        """rows: (ids, temperature, top_p, seed, cap, gen, truncated)."""
        lines = []
        for ids, t, p, seed, cap, gen, trunc in rows:
            e = {"key": self.key(ids, t, p, seed), "cap": cap, "gen": list(gen), "trunc": bool(trunc)}
            self.rows[e["key"]] = e
            lines.append(json.dumps(e, separators=(",", ":")))
        if lines:
            with self.path.open("a") as f:
                f.write("\n".join(lines) + "\n")

    def stats(self):
        return {"hits": self.hits, "misses": self.misses}


def cached_decode(cache, model, seqs, seeds, bs, max_new, eos, pad, budgets=None, deadline=None, sample=False,
                  temperature=0.8, top_p=0.95, decode_fn=None):
    """generate.decode through the cache. Rows found are returned as stored; the misses are generated in chunks of bs
    (in the given order; one sampling seed per chunk, derived from its rows' seeds) and written back. budgets (per row)
    caps each row below max_new; no new chunk starts after deadline (epoch s). Returns (toks, truncated) per row, None
    for rows not generated in time."""
    if decode_fn is None:
        from generate import decode as decode_fn
    t, p = (temperature, top_p) if sample else GREEDY
    caps = [max(1, b) for b in budgets] if budgets else [max_new] * len(seqs)
    out = [cache.get(s, t, p, sd, c) for s, sd, c in zip(seqs, seeds, caps)]
    miss = [i for i, o in enumerate(out) if o is None]
    cache.hits += len(seqs) - len(miss)
    for j in range(0, len(miss), bs):
        if deadline and time.time() > deadline:
            break
        rows = miss[j:j + bs]
        seed = int(hashlib.sha256(",".join(str(seeds[i]) for i in rows).encode()).hexdigest(), 16) % 2**31
        toks, _, _ = decode_fn(model, [seqs[i] for i in rows], max(caps[i] for i in rows), eos, pad, sample=sample,
                               seed=seed, temperature=temperature, top_p=top_p)
        new = []
        for i, row in zip(rows, toks):
            row = row[:caps[i]]
            out[i] = (row, not (row and row[-1] in eos))
            new.append((seqs[i], t, p, seeds[i], caps[i], row, out[i][1]))
        cache.put_many(new)
        cache.misses += len(rows)
        cache.gen_tokens += sum(len(r) for _, _, _, _, _, r, _ in new)
    return [o[0] if o else None for o in out], [o[1] if o else None for o in out]
