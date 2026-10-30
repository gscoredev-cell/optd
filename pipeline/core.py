"""Pure-Python helpers shared by the stages (no torch): scoring, boundaries, seeds, split, thresholds, status.

Everything here is covered by tests/test_core.py and runs on the local machine without a GPU.
"""
import hashlib
import json
import math
import os
import random
import re
import time
from fractions import Fraction

SYSTEM = ("Solve the problem step by step. Write each step on its own line. "
          "End with exactly one line: The final answer is <number>")
FINAL = "The final answer is"

_NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+"
_VALUE = re.compile(
    r"(?P<fneg>-?)\\d?frac\{\s*(?P<fn>" + _NUM + r")\s*\}\{\s*(?P<fd>" + _NUM + r")\s*\}"   # \frac{a}{b}
    r"|(?P<sn>" + _NUM + r")\s*/\s*(?P<sd>" + _NUM + r")(?![\d.])"                          # a/b
    r"|(?P<num>" + _NUM + r")")                                                             # plain number


def _frac(s):
    return Fraction(s.replace(",", ""))


def _value(m):
    """(value, decimals) for one regex match; decimals is None for exact forms (fractions, integers)."""
    try:
        if m.group("fn") is not None:
            v = _frac(m.group("fn")) / _frac(m.group("fd"))
            return (-v if m.group("fneg") else v), None
        if m.group("sn") is not None:
            return _frac(m.group("sn")) / _frac(m.group("sd")), None
    except ZeroDivisionError:
        return None
    s = m.group("num")
    return _frac(s), (len(s.split(".")[1]) if "." in s else None)


def extract_strict(text):
    """Number right after the last "The final answer is" (within 60 chars), else None (counts as wrong)."""
    i = text.rfind(FINAL)
    if i < 0:
        return None
    m = _VALUE.search(text[i + len(FINAL): i + len(FINAL) + 60])
    return _value(m) if m else None


def extract_lenient(text):
    """Strict answer, else the last number in the text."""
    v = extract_strict(text)
    if v is not None:
        return v
    last = None
    for m in _VALUE.finditer(text):
        last = m
    return _value(last) if last else None


def matches(pred, gold):
    """pred = (value, decimals) from extract_*; gold = Fraction. A decimal answer counts if it equals gold to the
    precision it was written with (2.33 for 7/3); exact forms must be equal."""
    if pred is None:
        return False
    v, dec = pred
    if dec is None:
        return v == gold
    return abs(v - gold) <= Fraction(1, 2 * 10 ** dec)


def gold_gsm(answer):
    """GSM answers end with '#### <number>'."""
    return _frac(answer.rsplit("####", 1)[1].strip())


def boundaries(token_texts, min_gap=8, max_n=12):
    """Indices of tokens whose decoded text contains a newline, at least min_gap apart, at most max_n (§9.1)."""
    out = []
    for i, t in enumerate(token_texts):
        if "\n" in t and (not out or i - out[-1] >= min_gap):
            out.append(i)
            if len(out) == max_n:
                break
    return out


def rollback_cut(bounds, s):
    """Prefix length (generated tokens kept) for a first flag at 1-based boundary s: tokens before boundary s-1,
    or 0 (restart from the prompt) when s == 1 (§9.3)."""
    return 0 if s == 1 else bounds[s - 2]


def seed_for(run_id, arm, problem_id):
    return int(hashlib.sha256(f"{run_id}:{arm}:{problem_id}".encode()).hexdigest(), 16) % 2**31


def split_by_oid(original_ids, seed=0):
    """Shuffle the unique original_ids with a fixed seed; first half -> train, second half -> test (§8.1)."""
    oids = sorted(set(original_ids))
    random.Random(seed).shuffle(oids)
    h = len(oids) // 2
    return set(oids[:h]), set(oids[h:])


def holdout_by_oid(original_ids, frac=0.2, seed=0):
    oids = sorted(set(original_ids))
    random.Random(seed).shuffle(oids)
    return set(oids[:max(1, round(frac * len(oids)))])


def tau_at_fpr(neg_scores, fpr=0.10):
    """Smallest threshold t with share(neg >= t) <= fpr, flagging is score >= t."""
    s = sorted(neg_scores, reverse=True)
    k = math.floor(fpr * len(s) + 1e-9)
    return s[-1] if k >= len(s) else math.nextafter(s[k], math.inf)


def auroc(pos_scores, neg_scores):
    """P(pos > neg) + 0.5 P(tie), by ranks."""
    allv = sorted([(v, 1) for v in pos_scores] + [(v, 0) for v in neg_scores])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2   # average 1-based rank of the tie block
        rank_sum += r * sum(lbl for _, lbl in allv[i:j])
        i = j
    p, n = len(pos_scores), len(neg_scores)
    return (rank_sum - p * (p + 1) / 2) / (p * n) if p and n else float("nan")


def write_status(run_dir, step, state="RUNNING", msg=""):
    """Atomic status.json (tmp file in the same directory + os.replace) and the .step file the heartbeat reads."""
    run_dir = str(run_dir)
    with open(os.path.join(run_dir, ".step"), "w") as f:
        f.write(step)
    tmp = os.path.join(run_dir, f".status.py.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump({"run_id": os.path.basename(run_dir.rstrip("/\\")), "state": state, "step": step, "msg": msg,
                   "ts": int(time.time())}, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(run_dir, "status.json"))


def kill_plan(steps):
    """(ckpt_every, kill_at) for the SMOKE kill/resume check: the spec's (10, 15) when there are more than 15 steps;
    with fewer steps a scaled plan whose only checkpoint before the kill is ckpt_every; None under 5 steps."""
    if steps > 15:
        return 10, 15
    if steps >= 5:
        c = (steps - 1) // 2
        return c, c + 1
    return None


def arm_metrics(rows):
    """rows: dicts with a0_ok, ok, flagged, a0_tokens, tokens, truncated. Per-arm numbers vs A0 (§9.5)."""
    n = len(rows)
    if not n:
        return {}
    rec = sum(1 for r in rows if not r["a0_ok"] and r["ok"])
    brk = sum(1 for r in rows if r["a0_ok"] and not r["ok"])
    a0_right = [r for r in rows if r["a0_ok"]]
    return {"n": n, "acc": sum(r["ok"] for r in rows) / n, "recovered": rec, "broken": brk, "net": rec - brk,
            "interv_rate": sum(r["flagged"] for r in rows) / n,
            "fpr": sum(r["flagged"] for r in a0_right) / len(a0_right) if a0_right else float("nan"),
            "extra_tokens": sum(r["tokens"] - r["a0_tokens"] for r in rows) / n,
            "trunc_rate": sum(r["truncated"] for r in rows) / n}
