"""Data (§8): GSM-Symbolic split by original_id, deterministic samples, RandomCalculation. No torch.

Problems are dicts {pid, set, oid, question, gold}; question text stays inside the pod (never written to results).
"""
import json
import pathlib
import random
import re
from fractions import Fraction

from core import gold_gsm, split_by_oid

CONFIGS = ("main", "p1", "p2")


def load_gsm(root):
    """{config: [row, ...]} from the dataset snapshot directory (main/test.jsonl, p1/..., p2/...)."""
    root = pathlib.Path(root)
    return {c: [json.loads(l) for l in (root / c / "test.jsonl").read_text().splitlines() if l.strip()]
            for c in CONFIGS}


def gsm_problem(config, row):
    return {"pid": f"{config}:{row['id']}:{row['instance']}", "set": config, "oid": row["original_id"],
            "question": row["question"], "gold": gold_gsm(row["answer"])}


def make_split(gsm, seed=0):
    """train/test original_ids over the union of all three configs (seed 0)."""
    return split_by_oid([r["original_id"] for c in CONFIGS for r in gsm[c]], seed)


def problems(gsm, configs, oids):
    return sorted((gsm_problem(c, r) for c in configs for r in gsm[c] if r["original_id"] in oids),
                  key=lambda p: p["pid"])


def sample(items, n, seed=0):
    """Deterministic sample of n items (sorted by pid first), returned sorted by pid."""
    items = sorted(items, key=lambda p: p["pid"])
    return sorted(random.Random(seed).sample(items, min(n, len(items))), key=lambda p: p["pid"])


# RandomCalculation (§8.2): a pure-Python port of wumingqi/LLM-Math-Evaluation random_calculation/
# generate_datasets.py. Same leaves (0-99, i/d for d in 2..9, i^2, i^3), operators + - * /, the same parenthesis rule
# (operands of * and / that contain an operator are parenthesised; + and - never are), and the answer is the exact value
# of the expression string as written, as the original computes it by parsing the string.
_LEAVES = ([(str(i), 0) for i in range(100)]
           + [(f"\\frac{{{i}}}{{{d}}}", 0) for d in range(2, 10) for i in range(1, 100)]
           + [(f"{i}^2", 0) for i in range(2, 100)] + [(f"{i}^3", 0) for i in range(2, 100)])


def _expr(rng, n_ops):
    if n_ops == 0:
        return rng.choice(_LEAVES)
    j = rng.randrange(n_ops)
    a, b = _expr(rng, j), _expr(rng, n_ops - 1 - j)
    op = rng.choice("+-*/")
    if rng.random() < 0.5:
        a, b = b, a
    (sa, ca), (sb, cb) = a, b
    if op in "*/":
        sa = f"({sa})" if ca >= 1 else sa
        sb = f"({sb})" if cb >= 1 else sb
    return f"{sa}{op}{sb}", ca + cb + 1


def eval_expr(s, steps=None):
    """Exact value of an expression string: + - * / ^int, parentheses, \\frac{a}{b}. Left-associative. With a list
    `steps`, every binary operation is appended as "a op b = v" in evaluation order (the reference solution's steps)."""
    toks = re.findall(r"\\frac|\d+|[-+*/^(){}]", s)
    pos = 0

    def note(a, op, b, v):
        if steps is not None:
            steps.append(f"{_fmt(a)} {op} {_fmt(b)} = {_fmt(v)}")

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(t=None):
        nonlocal pos
        tok = toks[pos]
        if t is not None and tok != t:
            raise ValueError(f"expected {t}, got {tok}")
        pos += 1
        return tok

    def base():
        t = peek()
        if t == "\\frac":
            take(); take("{"); n = expr(); take("}"); take("{"); d = expr(); take("}")
            note(n, "/", d, n / d)
            return n / d
        if t == "(":
            take(); v = expr(); take(")")
            return v
        if t == "{":
            take(); v = expr(); take("}")
            return v
        return Fraction(int(take()))

    def factor():
        v = base()
        if peek() == "^":
            take()
            e = int(take())
            note(v, "^", Fraction(e), v ** e)
            v = v ** e
        return v

    def term():
        v = factor()
        while peek() in ("*", "/"):
            op = take()
            w = factor()
            r = v * w if op == "*" else v / w
            note(v, op, w, r)
            v = r
        return v

    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take()
            w = term()
            r = v + w if op == "+" else v - w
            note(v, op, w, r)
            v = r
        return v

    v = expr()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in {s!r}")
    return v


def _fmt(v):
    """Fraction as an integer, an exact decimal (denominator 2^a 5^b, up to 12 places), or a/b."""
    if v.denominator == 1:
        return str(v.numerator)
    for k in range(1, 13):
        if (v * 10 ** k).denominator == 1:
            n = (v * 10 ** k).numerator
            s = f"{abs(n):0{k + 1}d}"
            return ("-" if n < 0 else "") + s[:-k] + "." + s[-k:]
    return f"{v.numerator}/{v.denominator}"


def rc_reference(p):
    """Reference solution text for a RandomCalculation problem: the evaluation steps and the result (SMOKE-8 teacher)."""
    steps = []
    v = eval_expr(p["expr"], steps)
    return "\n".join(steps + [f"The final answer is {_fmt(v)}"])


def random_calculation(n, seed=0, min_ops=4, max_ops=8):
    """n problems with 4-8 operations; skips division by zero and |value| > 1e12."""
    rng, out, i = random.Random(seed), [], 0
    while len(out) < n:
        s, ops = _expr(rng, rng.randint(min_ops, max_ops))
        try:
            v = eval_expr(s)
        except ZeroDivisionError:
            continue
        if abs(v) > 10**12:
            continue
        out.append({"pid": f"rc:{seed}:{i:04d}", "set": "rc", "oid": None, "ops": ops, "expr": s,
                    "question": f"Evaluate this LaTeX numerical expression: {s}", "gold": v})
        i += 1
    return out
