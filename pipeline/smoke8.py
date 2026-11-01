"""SMOKE-8 (Qwen3.5-9B self-teacher with privileged information; one pod, <= 150 min, $6). No pointer arms.

Z0  environment: transformers >= 5, flash-linear-attention + causal-conv1d importable, no "fast path is not available"
    warning while loading, bf16 matmul, template check for the two prompt variants (student / teacher with the
    reference line): both render with enable_thinking=False and share the user-turn suffix token for token.
Z1  difficulty calibration: student greedy on 60 problems each of p1, p2 (train templates) and RC at 3/4/5/6 ops;
    sets with accuracy in [35%, 70%] are kept.
Z2  data on the kept sets: 4 samples (T 0.8, cap 800) per train and test problem through the cache, greedy test
    traces; contested = 1..3 of 4 correct. Test units = wrong greedy traces of contested test problems.
Z3  self-teacher: the same weights with a system line "Reference solution: <dataset answer>" (GSM answer field; the
    evaluation steps for RandomCalculation). Teacher greedy accuracy with the reference; mean teacher - student
    log-prob gap per token on correct vs wrong student traces.
Z4  training units = contested train problems (restart from the prompt): 4 continuations per unit with the adapter on
    over the whole sequence; rkl with PPO ratio clipping (eps 0.2), lr 5e-5; dev every 10 steps on 30 traces (split by
    problem, stratified by set, never more than 30% of a set); up to 3 rounds, best checkpoint kept; kill/resume once.
Z5  arms on the test units: E0 plain resample, E2 resample with the hint line, E6 resample with the adapter, E6h adapter
    + hint; identical batches and seeds. Harm: adapter vs plain resample on 60 correct greedy test traces.
Z6  benchmark number: greedy pass@1 on all test problems, base vs adapter always on (A6), per set.
Z7  report + FULL projection.

Generation and log-probs use plain HF calls (no prefix cache): the hybrid model's cache object is not handed between
forwards. Every adapter-off generation goes through the generation cache. Question text never leaves the pod."""
import collections
import json
import logging
import pathlib
import random
import statistics
import time

import oracle as O
from core import SYSTEM, extract_strict, matches, primary_ok, seed_for
from gencache import GenCache, cached_decode

MAX_NEW = 800
HINT = "\nWait — let me re-check that step.\n"
REF_LINE = "\nReference solution:\n"
CAL_N, CAL_LO, CAL_HI = 60, 0.35, 0.70
TRAIN_TOTAL, TEST_TOTAL, DEV_N, DEV_MAX_SHARE, HARM_N, CONT_K, ROUNDS = 400, 240, 30, 0.30, 60, 4, 3
LR, CLIP, GAP_N = 5e-5, 0.2, 200
RC_OPS = (3, 4, 5, 6)
RC_SEEDS = {"cal": 800, "train": 810, "test": 820}      # + ops; earlier seeds: 0, 103-105, 200, 503/504, 603/604, 703
NS = "optE8"
# minutes (A100 estimates x 1.3) and the least time a stage may start with
BUDGET_MIN = {"Z0": 8, "Z1": 8, "Z2": 30, "Z3": 8, "Z4": 35, "Z5": 14, "Z6": 8, "Z7": 2}
START_MIN = {"Z0": 3, "Z1": 3, "Z2": 8, "Z3": 3, "Z4": 8, "Z5": 5, "Z6": 3, "Z7": 1}
ORDER = ["Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"]
SKIP = "skipped: not enough pod time left"
ARMS = ("E0", "E2", "E6", "E6h")


# ---- pure helpers ----------------------------------------------------------------------------------------------------

def choose_sets(acc, lo=CAL_LO, hi=CAL_HI):
    """Sets whose calibration accuracy lies in [lo, hi], in the given order."""
    return [s for s, a in acc.items() if a is not None and lo <= a <= hi]


def split_sizes(total, sets):
    return {s: total // len(sets) for s in sets} if sets else {}


def dev_split(units, n=DEV_N, max_share=DEV_MAX_SHARE, seed=60):
    """Dev problems: n units stratified by set in proportion to the set's share, never more than max_share of any set
    (a scarce stratum keeps most of its units for training). Returns (dev units, train units), disjoint by problem."""
    by = collections.defaultdict(list)
    for u in units:
        by[u["set"]].append(u)
    rng = random.Random(seed)
    dev = []
    for s in sorted(by):
        want = round(n * len(by[s]) / max(len(units), 1))
        take = min(want, int(max_share * len(by[s])))
        dev += rng.sample(sorted(by[s], key=lambda u: u["pid"]), take)
    groups = {u.get("group", u["pid"]) for u in dev}            # disjoint by original problem, like the test split
    return dev, [u for u in units if u.get("group", u["pid"]) not in groups]


def teacher_system(ref):
    return SYSTEM + REF_LINE + ref


def common_suffix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def arm_stats(rows, groups):
    """Recovery per arm; deltas vs E0 and E6h - E6 with cluster-bootstrap CIs (10,000 resamples by original_id)."""
    out = {"n": len(rows), "problems": len(set(groups))}
    for a in ARMS:
        out[f"rec_{a}"] = O.recovery(rows, a)
        out[f"tokens_{a}"] = statistics.mean(r[f"tok_{a}"] for r in rows) if rows else float("nan")
        out[f"trunc_{a}"] = statistics.mean(r[f"trunc_{a}"] for r in rows) if rows else float("nan")
    if not rows:
        return out
    for name, a, b in (("e6_minus_e0", "E6", "E0"), ("e2_minus_e0", "E2", "E0"), ("e6h_minus_e0", "E6h", "E0"),
                       ("e6h_minus_e6", "E6h", "E6"), ("e6_minus_e2", "E6", "E2")):
        point, ci = O.cluster_bootstrap(rows, lambda rs, a=a, b=b: O.delta(rs, a, b), groups)
        out[name], out[f"{name}_ci"] = point, ci
    return out


def contested(ts):
    return 0 < sum(t["ok"] for t in ts) < len(ts)


# ---- model helpers (torch) -------------------------------------------------------------------------------------------

class WarnCatch(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage()[:300])


def load_qwen35(repo):
    """The multimodal checkpoint used as a text-only causal LM; returns (model, class name, warnings while loading)."""
    import torch
    import transformers
    from generate import model_dir
    catch = WarnCatch()
    lg = logging.getLogger("transformers")
    lg.addHandler(catch)
    try:
        try:
            cls = transformers.AutoModelForImageTextToText
            model = cls.from_pretrained(model_dir(repo), dtype=torch.bfloat16, attn_implementation="sdpa")
        except Exception as e:                                  # text-only class as a fallback
            catch.msgs.append(f"ImageTextToText load failed: {type(e).__name__}: {str(e)[:150]}")
            cls = transformers.AutoModelForCausalLM
            model = cls.from_pretrained(model_dir(repo), dtype=torch.bfloat16, attn_implementation="sdpa")
    finally:
        lg.removeHandler(catch)
    return model.to("cuda").eval(), type(model).__name__, catch.msgs


def eos_ids(model, tok):
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, list) else [e]) if e is not None else set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def gen8(model, seqs, max_new, eos, pad, sample=False, seed=0, temperature=0.8, top_p=0.95, adapter_on=False):
    """Plain HF generate over left-padded sequences (adapter on/off for the whole sequence). Returns
    (new tokens incl. the first EOS, truncated, None) like generate.decode."""
    import torch
    from generate import _left_pad, adapter
    ids, mask, _ = _left_pad(seqs, pad, model.device)
    kw = (dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if sample
          else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    if sample:
        torch.manual_seed(seed)
    with torch.no_grad(), adapter(model, adapter_on):
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, eos_token_id=sorted(eos),
                             pad_token_id=pad, **kw)
    toks, trunc = [], []
    for row in out[:, ids.shape[1]:].tolist():
        k = next((i for i, t in enumerate(row) if t in eos), None)
        toks.append(row[:k + 1] if k is not None else row)
        trunc.append(k is None)
    return toks, trunc, None


def gen8_batches(model, seqs, bs, max_new, eos, pad, seed_fn=None, budgets=None, deadline=None, **kw):
    """gen8 over chunks of bs rows with one sampling seed per chunk; rows past the deadline are missing."""
    toks, trunc = [], []
    for c, i in enumerate(range(0, len(seqs), bs)):
        if deadline and time.time() > deadline:
            break
        chunk = seqs[i:i + bs]
        b = [max(1, x) for x in budgets[i:i + bs]] if budgets else [max_new] * len(chunk)
        t, _, _ = gen8(model, chunk, max(b), eos, pad, seed=seed_fn(c) if seed_fn else 0, **kw)
        for row, lim in zip(t, b):
            cut = row[:lim]
            toks.append(cut)
            trunc.append(not (cut and cut[-1] in eos))
    return toks, trunc


def logprobs8(model, prefixes, conts, bs, adapter_on=False):
    """log p(cont_t | prefix, cont_<t) with one full forward per row (no position ids: the model derives them)."""
    import torch
    from generate import _left_pad, adapter
    out = []
    for i in range(0, len(prefixes), bs):
        seqs = [p + c for p, c in zip(prefixes[i:i + bs], conts[i:i + bs])]
        ids, mask, _ = _left_pad(seqs, 0, model.device)
        with torch.no_grad(), adapter(model, adapter_on):
            logits = model(input_ids=ids, attention_mask=mask).logits
        width = ids.shape[1]
        for r, c in enumerate(conts[i:i + bs]):
            start = width - len(c) - 1
            lp = logits[r, start:start + len(c)].float().log_softmax(-1)
            out.append(lp.gather(-1, torch.tensor(c, dtype=torch.long, device=lp.device)[:, None]).squeeze(-1).cpu())
        del logits
    return out


def lora_targets(model):
    """Names of the Linear modules of the language model's decoder layers (never lm_head or the vision tower)."""
    import torch
    names = [n for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)
             and ".layers." in n and "visual" not in n and "lm_head" not in n and "mtp" not in n]
    return names or "all-linear"


# ---- context ---------------------------------------------------------------------------------------------------------

class Ctx8:
    def __init__(self, cfg, tiers, tier, run_id, run_dir):
        import data as D
        from generate import load_tokenizer, prompt_ids
        from huggingface_hub import snapshot_download
        self.cfg, self.run_id, self.run_dir, self.tier = cfg, run_id, run_dir, tier
        self.repo = cfg["smoke8_model"]
        self.gen_bs = {80: 64, 48: 32, 32: 16, 24: 8}[tier]
        self.lp_bs = {80: 16, 48: 8, 32: 4, 24: 4}[tier]
        self.micro = 4 if tier == 80 else 2
        self.tok = load_tokenizer(self.repo)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._model, self._lora, self.eos = None, False, None
        self.cache = GenCache(self.repo)
        gsm = D.load_gsm(snapshot_download(cfg["dataset"], repo_type="dataset", local_files_only=True))
        self.train_oids, self.test_oids = D.make_split(gsm)
        self.answers = {f"{c}:{r['id']}:{r['instance']}": r["answer"] for c in D.CONFIGS for r in gsm[c]}
        self.gsm = gsm
        self.probs = {}            # pid -> problem (question, gold, set, oid, ref)
        self.hint_ids = self.tok.encode(HINT, add_special_tokens=False)
        self._prompt_ids = prompt_ids

    # problems
    def gsm_pool(self, config, split):
        import data as D
        return D.problems(self.gsm, [config], self.train_oids if split == "train" else self.test_oids)

    def rc_pool(self, ops, n, seed):
        import data as D
        return [dict(p, pid=f"rc{ops}:{p['pid'].split(':', 1)[1]}", set=f"rc{ops}")
                for p in D.random_calculation(n, seed=seed, min_ops=ops, max_ops=ops)]

    def register(self, probs):
        import data as D
        for p in probs:
            p = dict(p)
            p["ref"] = self.answers[p["pid"]] if p["set"] in D.CONFIGS else D.rc_reference(p)
            self.probs[p["pid"]] = p
        return [self.probs[p["pid"]] for p in probs]

    def group(self, pid):
        p = self.probs[pid]
        return f"gsm:{p['oid']}" if p.get("oid") is not None else pid

    def student_prompt(self, pid):
        return self._prompt_ids(self.tok, self.probs[pid]["question"])

    def teacher_prompt(self, pid):
        p = self.probs[pid]
        return list(self.tok.apply_chat_template([{"role": "system", "content": teacher_system(p["ref"])},
                                                  {"role": "user", "content": p["question"]}],
                                                 tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                                 return_dict=False))

    # model
    def model(self):
        if self._model is None:
            self._model, self.model_class, self.load_warnings = load_qwen35(self.repo)
            self.eos = eos_ids(self._model, self.tok)
        return self._model

    def lora(self):
        import torch
        from train_repair import lora_state, set_lora
        if not self._lora:
            from peft import LoraConfig, get_peft_model
            m = self.model()
            torch.manual_seed(seed_for(self.run_id, "lora", 0))
            self._model = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                       target_modules=lora_targets(m)))
            init = self.run_dir / "stage_Z4" / "lora_init.pt"
            init.parent.mkdir(exist_ok=True)
            if init.exists():
                set_lora(self._model, torch.load(init, weights_only=False))
            else:
                torch.save(lora_state(self._model), init)
            self._lora = True
        return self._model

    def lora_init(self):
        import torch
        return torch.load(self.run_dir / "stage_Z4" / "lora_init.pt", weights_only=False)

    # time and artifacts
    def budget(self, name):
        from smoke3 import pipeline_deadline
        later = ORDER[ORDER.index(name) + 1:]
        end = pipeline_deadline() - sum(BUDGET_MIN[r] for r in later) * 60
        return time.time() + START_MIN[name] * 60 <= end, min(time.time() + BUDGET_MIN[name] * 60, end)

    def save(self, d, name, obj):
        import torch
        torch.save(obj, d / f"{name}.pt")

    def load(self, stage, name):
        import torch
        return torch.load(self.run_dir / f"stage_{stage}" / f"{name}.pt", weights_only=False)

    def has(self, stage, name):
        return (self.run_dir / f"stage_{stage}" / f"{name}.pt").exists()

    # generation
    def gen(self, seqs, seeds, deadline, sample=True, budgets=None):
        """Adapter-off generation through the cache (plain generate)."""
        model = self.model()
        return cached_decode(self.cache, model, seqs, seeds, self.gen_bs, MAX_NEW, self.eos, self.pad, budgets=budgets,
                             deadline=deadline, sample=sample, decode_fn=gen8)

    def gen_adapter(self, seqs, seeds, deadline, sample=True, budgets=None):
        """Adapter-on generation (never cached); seeds[c] per chunk."""
        return gen8_batches(self.lora(), seqs, self.gen_bs, MAX_NEW, self.eos, self.pad, seed_fn=lambda c: seeds[c],
                            budgets=budgets, deadline=deadline, sample=sample, adapter_on=True)

    def trace(self, pid, prompt, gen, trunc, k="g"):
        text = self.tok.decode(gen, skip_special_tokens=True)
        g = self.probs[pid]["gold"]
        return {"pid": pid, "set": self.probs[pid]["set"], "prompt": prompt, "gen": gen, "truncated": trunc, "k": k,
                "ok": primary_ok(text, g, trunc), "ok_strict": matches(extract_strict(text), g)}

    def ok(self, pid, gen, trunc):
        return primary_ok(self.tok.decode(gen, skip_special_tokens=True), self.probs[pid]["gold"], trunc)

    def mark(self):
        return self.cache.hits, self.cache.misses, self.cache.gen_tokens, time.time()

    def since(self, mark, tag):
        h, mi, tok, t0 = mark
        sec = max(time.time() - t0, 1e-9)
        return {f"{tag}_cache_hits": self.cache.hits - h, f"{tag}_cache_misses": self.cache.misses - mi,
                f"{tag}_tok_per_s": (self.cache.gen_tokens - tok) / sec, f"{tag}_sec": sec}


def _rate(xs):
    return statistics.mean(xs) if xs else float("nan")


# ---- stages ----------------------------------------------------------------------------------------------------------

def z0(c8, d):
    import importlib
    import torch
    import transformers
    ok, _ = c8.budget("Z0")
    if not ok:
        return {"Z0_skipped": SKIP}
    m = {"z0_transformers": transformers.__version__, "z0_transformers_ge5": int(transformers.__version__.split(".")[0]) >= 5}
    for mod in ("fla", "causal_conv1d", "triton", "peft"):
        try:
            x = importlib.import_module(mod)
            m[f"z0_{mod}"] = getattr(x, "__version__", "ok")
        except Exception as e:
            m[f"z0_{mod}"] = f"missing ({type(e).__name__})"
    t0 = time.time()
    model = c8.model()
    m.update({"z0_load_sec": time.time() - t0, "z0_model_class": c8.model_class, "z0_load_warnings": c8.load_warnings,
              "z0_fast_path_warning": any("fast path" in w.lower() for w in c8.load_warnings),
              "z0_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
              "z0_vram_gb_after_load": torch.cuda.memory_allocated() / 2**30, "z0_eos": sorted(c8.eos)})
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        m["z0_module_fast_path_flags"] = {k: bool(getattr(mq, k)) for k in dir(mq) if "fast_path" in k.lower()}
    except Exception as e:
        m["z0_module_fast_path_flags"] = f"n/a ({type(e).__name__})"
    # template check: student vs teacher variant on 20 train p1 problems
    probs = c8.register(c8.gsm_pool("p1", "train")[:20])
    suf, think, hint_rt = [], 0, c8.tok.decode(c8.hint_ids) == HINT
    for p in probs:
        s, t = c8.student_prompt(p["pid"]), c8.teacher_prompt(p["pid"])
        suf.append(common_suffix(s, t))
        txt = c8.tok.decode(s)
        think += txt.rstrip().endswith("</think>") or "<think>" in txt[-40:]
    user_len = [len(c8.tok.encode(p["question"], add_special_tokens=False)) for p in probs]
    m.update({"z0_template_n": 20, "z0_common_suffix_min": min(suf), "z0_user_tokens_mean": statistics.mean(user_len),
              "z0_suffix_covers_user_turn": sum(k >= u + 3 for k, u in zip(suf, user_len)),
              "z0_think_block": think, "z0_hint_roundtrip": hint_rt,
              "z0_prompt_tail": repr(c8.tok.decode(c8.student_prompt(probs[0]["pid"])[-12:]))})
    # one tiny generation to confirm the path works end to end
    t0 = time.time()
    toks, trunc, _ = gen8(model, [c8.student_prompt(probs[0]["pid"])], 32, c8.eos, c8.pad)
    m.update({"z0_gen_smoke_tokens": len(toks[0]), "z0_gen_smoke_sec": time.time() - t0})
    return m


def z1(c8, d):
    ok, deadline = c8.budget("Z1")
    if not ok:
        return {"Z1_skipped": SKIP}
    import data as D
    sets = {"p1": c8.register(D.sample(c8.gsm_pool("p1", "train"), CAL_N, seed=8)),
            "p2": c8.register(D.sample(c8.gsm_pool("p2", "train"), CAL_N, seed=8))}
    for k in RC_OPS:
        sets[f"rc{k}"] = c8.register(c8.rc_pool(k, CAL_N, RC_SEEDS["cal"] + k))
    order = [p for s in sets for p in sets[s]]
    order = random.Random(8).sample(order, len(order))          # a time cut keeps every set represented
    mk = c8.mark()
    toks, trunc = c8.gen([c8.student_prompt(p["pid"]) for p in order], [0] * len(order), deadline, sample=False)
    m = c8.since(mk, "z1")
    traces = [c8.trace(p["pid"], c8.student_prompt(p["pid"]), tk, tr) for p, tk, tr in zip(order, toks, trunc) if tk is not None]
    c8.save(d, "traces", traces)
    acc, cal = {}, {}
    for s in sets:
        ts = [t for t in traces if t["set"] == s]
        acc[s] = _rate([t["ok"] for t in ts]) if len(ts) >= CAL_N // 2 else None
        cal[s] = {"n": len(ts), "acc": acc[s], "strict": _rate([t["ok_strict"] for t in ts]),
                  "trunc": _rate([t["truncated"] for t in ts]),
                  "mean_tokens": _rate([len(t["gen"]) for t in ts])}
    chosen = choose_sets(acc)
    c8.save(d, "chosen", chosen)
    return {**m, "z1_cal": cal, "z1_chosen": chosen, "z1_rule": f"accuracy in [{CAL_LO}, {CAL_HI}]"}


def select_problems(c8, chosen):
    """The Z2 problem list for the kept sets (deterministic; registers them with the context)."""
    import data as D
    n_tr, n_te = split_sizes(TRAIN_TOTAL, chosen), split_sizes(TEST_TOTAL, chosen)
    probs = []
    for s in chosen:
        if s.startswith("rc"):
            k = int(s[2:])
            tr = c8.rc_pool(k, n_tr[s], RC_SEEDS["train"] + k)
            te = c8.rc_pool(k, n_te[s], RC_SEEDS["test"] + k)
        else:
            tr = D.sample(c8.gsm_pool(s, "train"), n_tr[s], seed=82)
            te = D.sample(c8.gsm_pool(s, "test"), n_te[s], seed=82)
        probs += [dict(p, split="train") for p in c8.register(tr)] + [dict(p, split="test") for p in c8.register(te)]
    return random.Random(82).sample(probs, len(probs)), n_tr, n_te


def _restore(c8):
    """After a pod restart the context has no problems: re-register Z2's list (same seeds) so prompts can be rebuilt."""
    if c8.has("Z1", "chosen"):
        select_problems(c8, c8.load("Z1", "chosen"))            # idempotent


def z2(c8, d):
    ok, deadline = c8.budget("Z2")
    if not ok or not c8.has("Z1", "chosen"):
        return {"Z2_skipped": SKIP if not ok else "skipped: no calibration"}
    chosen = c8.load("Z1", "chosen")
    if not chosen:
        return {"Z2_skipped": "skipped: no set in the accuracy band"}
    probs, n_tr, n_te = select_problems(c8, chosen)
    split = {p["pid"]: p["split"] for p in probs}
    mk = c8.mark()
    rows = [(p["pid"], k) for p in probs for k in range(4)] + [(p["pid"], "g") for p in probs if p["split"] == "test"]
    sampled = [r for r in rows if r[1] != "g"]
    greedy = [r for r in rows if r[1] == "g"]
    out = {}
    toks, trunc = c8.gen([c8.student_prompt(p) for p, _ in greedy], [0] * len(greedy), deadline, sample=False)
    out.update({r: (t, tr) for r, t, tr in zip(greedy, toks, trunc) if t is not None})
    toks, trunc = c8.gen([c8.student_prompt(p) for p, _ in sampled], [seed_for(NS, "s", f"{p}#{k}") for p, k in sampled], deadline)
    out.update({r: (t, tr) for r, t, tr in zip(sampled, toks, trunc) if t is not None})
    m = c8.since(mk, "z2")
    m["z2_rows_done"] = f"{len(out)}/{len(rows)}"
    if len(out) < len(rows):
        m["z2_cut"] = f"deadline: {len(rows) - len(out)} of {len(rows)} rows not generated"
    traces = [dict(c8.trace(p, c8.student_prompt(p), *out[(p, k)], k=k), split=split[p])
              for p in dict.fromkeys(r[0] for r in rows) if all((p, j) in out for j in range(4)) for k in range(4)]
    gtest = [dict(c8.trace(p, c8.student_prompt(p), *out[(p, "g")]), split="test")
             for p in dict.fromkeys(r[0] for r in greedy) if (p, "g") in out]
    c8.save(d, "samples", traces)
    c8.save(d, "greedy_test", gtest)
    c8.save(d, "problems", {p["pid"]: {"set": p["set"], "split": p["split"], "gold": str(c8.probs[p["pid"]]["gold"])}
                            for p in probs})      # ids and gold only: question text stays out of the artifacts
    by = collections.defaultdict(list)
    for t in traces:
        by[t["pid"]].append(t)
    cont = {p for p, ts in by.items() if contested(ts)}
    gw = {t["pid"]: t for t in gtest}
    units = sorted([gw[p] for p in cont if split[p] == "test" and p in gw and not gw[p]["ok"] and gw[p]["gen"]],
                   key=lambda t: t["pid"])
    correct = sorted([t for t in gtest if t["ok"]], key=lambda t: t["pid"])
    harm = sorted(random.Random(5).sample(correct, min(HARM_N, len(correct))), key=lambda t: t["pid"])
    c8.save(d, "train_units", sorted([{"pid": p, "set": by[p][0]["set"], "group": c8.group(p)} for p in cont if split[p] == "train"],
                                     key=lambda u: u["pid"]))
    c8.save(d, "test_units", units)
    c8.save(d, "harm", harm)
    m.update({"z2_problems": len(by), "z2_of": len(probs), "z2_sizes_train": n_tr, "z2_sizes_test": n_te,
              "z2_trunc": _rate([t["truncated"] for t in traces]), "z2_mean_tokens": _rate([len(t["gen"]) for t in traces]),
              "z2_test_units": len(units), "z2_harm_traces": len(harm), "z2_greedy_test": len(gtest)})
    for sp in ("train", "test"):
        for s in chosen + ["all"]:
            ps = [p for p in by if split[p] == sp and (s == "all" or by[p][0]["set"] == s)]
            hist = collections.Counter(sum(t["ok"] for t in by[p]) for p in ps)
            m[f"z2_{sp}_{s}"] = {"problems": len(ps), "hist": [hist.get(k, 0) for k in range(5)],
                                 "acc": _rate([t["ok"] for p in ps for t in by[p]]),
                                 "contested": _rate([p in cont for p in ps]),
                                 **({"greedy_acc": _rate([gw[p]["ok"] for p in ps if p in gw]),
                                     "wrong_greedy_contested": sum(1 for p in ps if p in cont and p in gw and not gw[p]["ok"])}
                                    if sp == "test" else {})}
    return m


def z3(c8, d):
    """Self-teacher: greedy with the reference on contested train problems; log-prob gap on sampled traces."""
    ok, deadline = c8.budget("Z3")
    if not ok or not c8.has("Z2", "train_units"):
        return {"Z3_skipped": SKIP if not ok else "skipped: no Z2 data"}
    _restore(c8)
    units = c8.load("Z2", "train_units")
    tcache = GenCache(c8.repo + "#teacher")
    mk = c8.mark()
    toks, trunc = cached_decode(tcache, c8.model(), [c8.teacher_prompt(u["pid"]) for u in units], [0] * len(units),
                                c8.gen_bs, MAX_NEW, c8.eos, c8.pad, deadline=deadline, sample=False, decode_fn=gen8)
    tea = {u["pid"]: c8.ok(u["pid"], tk, tr) for u, tk, tr in zip(units, toks, trunc) if tk is not None}
    c8.save(d, "teacher_ok", tea)
    m = {"z3_teacher_n": len(tea), "z3_teacher_acc": _rate(list(tea.values())),
         "z3_teacher_tok_per_s": sum(len(t) for t in toks if t) / max(time.time() - mk[3], 1e-9)}
    for s in sorted({u["set"] for u in units}):
        m[f"z3_teacher_acc_{s}"] = _rate([tea[u["pid"]] for u in units if u["pid"] in tea and u["set"] == s])
    # log-prob gap on up to GAP_N sampled train traces, balanced correct / wrong
    samples = [t for t in c8.load("Z2", "samples") if t["split"] == "train" and t["gen"]]
    rng = random.Random(3)
    ok_t = rng.sample([t for t in samples if t["ok"]], min(GAP_N // 2, sum(t["ok"] for t in samples)))
    bad_t = rng.sample([t for t in samples if not t["ok"]], min(GAP_N // 2, sum(not t["ok"] for t in samples)))
    sel = ok_t + bad_t
    if sel and time.time() < deadline:
        t0 = time.time()
        lp_t = logprobs8(c8.model(), [c8.teacher_prompt(t["pid"]) for t in sel], [t["gen"] for t in sel], c8.lp_bs)
        lp_s = logprobs8(c8.model(), [t["prompt"] for t in sel], [t["gen"] for t in sel], c8.lp_bs)
        sec = time.time() - t0
        gap = [float((a - b).mean()) for a, b in zip(lp_t, lp_s)]
        m.update({"z3_gap_n_correct": len(ok_t), "z3_gap_n_wrong": len(bad_t),
                  "z3_gap_correct": _rate(gap[:len(ok_t)]), "z3_gap_wrong": _rate(gap[len(ok_t):]),
                  "z3_student_logp_correct": _rate([float(b.mean()) for b in lp_s[:len(ok_t)]]),
                  "z3_student_logp_wrong": _rate([float(b.mean()) for b in lp_s[len(ok_t):]]),
                  "z3_prefill_tok_per_s": 2 * sum(len(t["prompt"]) + len(t["gen"]) for t in sel) / max(sec, 1e-9)})
    return m


def _round8(c8, units, tag, deadline, m):
    """CONT_K continuations per unit from the prompt with the adapter on; lp_s (adapter on), lp_t (teacher prompt,
    adapter off); teacher-also-wrong from Z3. Returns training examples."""
    model = c8.lora()
    rows = [(u, j) for u in units for j in range(CONT_K)]
    seqs = [c8.student_prompt(u["pid"]) for u, _ in rows]
    t0 = time.time()
    seeds = O.chunk_seeds(c8.run_id, f"cont{tag}", len(rows), c8.gen_bs)
    toks, trunc = c8.gen_adapter(seqs, seeds, deadline)
    gen_sec = time.time() - t0
    rows, seqs = rows[:len(toks)], seqs[:len(toks)]
    ex = [{"prefix": s, "cont": tk, "pid": u["pid"], "set": u["set"], "ok": c8.ok(u["pid"], tk, tr), "truncated": tr}
          for (u, _), s, tk, tr in zip(rows, seqs, toks, trunc) if tk]
    lp_s = logprobs8(model, [e["prefix"] for e in ex], [e["cont"] for e in ex], c8.lp_bs, adapter_on=True)
    t1 = time.time()
    lp_t = logprobs8(model, [c8.teacher_prompt(e["pid"]) for e in ex], [e["cont"] for e in ex], c8.lp_bs, adapter_on=False)
    prefill_sec = time.time() - t1
    for e, s, t in zip(ex, lp_s, lp_t):
        e["lp_s"], e["lp_t"] = s, t
    tea = c8.load("Z3", "teacher_ok") if c8.has("Z3", "teacher_ok") else {}
    adv = [float((-(e["lp_s"] - e["lp_t"])).clamp(-5, 5).mean()) for e in ex]
    m.update({f"{tag}_units": len(units), f"{tag}_continuations": len(ex), f"{tag}_cont_primary": _rate([e["ok"] for e in ex]),
              f"{tag}_cont_trunc": _rate([e["truncated"] for e in ex]), f"{tag}_gen_sec": gen_sec,
              f"{tag}_gen_tok_per_s": sum(len(e["cont"]) for e in ex) / max(gen_sec, 1e-9),
              f"{tag}_prefill_tok_per_s": sum(len(e["prefix"]) + len(e["cont"]) for e in ex) / max(prefill_sec, 1e-9),
              f"{tag}_teacher_also_wrong": _rate([not tea[u["pid"]] for u in units if u["pid"] in tea]),
              f"{tag}_mean_adv": _rate(adv)})
    return ex


def _dev_fn(c8, dev):
    seqs = [c8.student_prompt(u["pid"]) for u in dev]
    seeds = O.chunk_seeds(c8.run_id, "dev", len(seqs), c8.gen_bs)

    def fn(model):
        toks, trunc = gen8_batches(model, seqs, c8.gen_bs, MAX_NEW, c8.eos, c8.pad, seed_fn=lambda c: seeds[c],
                                   sample=True, adapter_on=True)
        return _rate([c8.ok(u["pid"], tk, tr) for u, tk, tr in zip(dev, toks, trunc)])
    return fn


def z4(c8, d):
    import math
    import shutil
    import torch
    from core import kill_plan
    from train_repair import _loss_plain, set_lora, train
    ok, deadline = c8.budget("Z4")
    _restore(c8)
    if not ok or not c8.has("Z2", "train_units"):
        return {"Z4_skipped": SKIP if not ok else "skipped: no Z2 data"}
    units = c8.load("Z2", "train_units")
    dev, train_units = dev_split(units)
    c8.save(d, "dev", dev)
    c8.save(d, "train_units", train_units)
    model = c8.lora()
    set_lora(model, c8.lora_init())
    m = {"z4_units": len(train_units), "z4_dev": len(dev), "z4_dev_by_set": dict(collections.Counter(u["set"] for u in dev)),
         "z4_units_by_set": dict(collections.Counter(u["set"] for u in train_units)),
         "adapter_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "z4_lr": LR, "z4_clip": CLIP,
         "z4_micro": c8.micro}
    eval_fn = _dev_fn(c8, dev)
    best, curves, t0, eval_sec, gen_sec, executed = None, [], time.time(), 0.0, 0.0, 0
    for rnd in range(1, ROUNDS + 1):
        if rnd > 1 and time.time() + 300 > deadline:
            m[f"round{rnd}_skipped"] = "stage time budget"
            break
        if rnd > 1:
            set_lora(model, best["state"])
        t1 = time.time()
        ex = _round8(c8, train_units, f"z4r{rnd}", deadline, m)
        gen_sec += time.time() - t1
        c8.save(d, f"examples_r{rnd}", [{k: v for k, v in e.items() if k not in ("prefix",)} for e in ex])
        if not ex or time.time() + 120 > deadline:
            m[f"round{rnd}_skipped"] = "no examples" if not ex else "stage time budget (continuations sampled, training skipped)"
            break
        ck = d / f"ckpt_r{rnd}"
        shutil.rmtree(ck, ignore_errors=True)
        steps = math.ceil(len(ex) / 16)
        early = {}
        kw = dict(loss_kind="rkl", lr=LR, clip_eps=CLIP, eval_fn=eval_fn, early=early, deadline=deadline - 60,
                  loss_fn=_loss_plain, micro=c8.micro)
        plan = kill_plan(steps) if rnd == 1 else None
        if plan:
            c, k = plan
            first, _ = train(model, ex, ck, c8.run_id, stop_at=k, ckpt_every=c, **kw)
            set_lora(model, c8.lora_init())
            losses, _ = train(model, ex, ck, c8.run_id, resume=True, ckpt_every=c, **kw)
            gap = [abs(first[s] - losses[s]) / max(1.0, abs(first[s])) for s in range(c + 1, k + 1) if s in losses and s in first]
            m.update({"train_ckpt_every": c, "train_kill_at": k, "resume_max_rel_diff": max(gap, default=float("nan")),
                      "resume_ok": bool(gap) and max(gap) <= 1e-3, "resume_exact": bool(gap) and max(gap) == 0.0})
            executed += k
        else:
            losses, _ = train(model, ex, ck, c8.run_id, ckpt_every=10, **kw)
            if rnd == 1:
                m.update({"resume_ok": None, "train_blocker": f"only {steps} steps: kill/resume check needs >= 5"})
        eval_sec += early.get("eval_sec", 0.0)
        executed += early.get("last_step", 0) - (c if plan else 0)
        curves.append({"round": rnd, "steps": steps, "examples": len(ex), "curve": early.get("curve", []),
                       "best_step": early.get("best_step"), "best_dev": early.get("best_score"), "stopped": early.get("stopped"),
                       "cut": early.get("cut", False), "loss_start": losses.get(1), "loss_end": losses.get(max(losses)) if losses else None})
        improved = early.get("best_state") is not None and (best is None or early["best_score"] > best["dev"])
        if improved:
            best = {"dev": early["best_score"], "step": early.get("best_step"), "round": rnd, "state": early["best_state"]}
        elif best is not None:
            m["rounds_stopped"] = f"round {rnd} did not improve dev"
            break
    sec = time.time() - t0
    if best is None:                       # nothing trained: no adapter.pt, so Z5/Z6 report the adapter as untrained
        best = {"dev": None, "step": 0, "round": 0, "state": c8.lora_init()}
        m["train_blocker"] = m.get("train_blocker") or "no round produced a checkpoint"
    else:
        torch.save(best["state"], d / "adapter.pt")
    set_lora(model, best["state"])
    pure = max(sec - eval_sec - gen_sec, 1e-9)
    m.update({"train_rounds": curves, "train_rounds_done": len(curves), "train_best_dev": best["dev"], "train_best_step": best["step"],
              "train_best_round": best["round"], "early_stop_fired": any(c["stopped"] for c in curves), "train_sec": sec,
              "dev_eval_sec": eval_sec, "resample_sec": gen_sec, "executed_steps": executed,
              "train_steps_per_s": executed / max(sec, 1e-9), "pure_train_steps_per_s": executed / pure})
    return m


def z5(c8, d):
    from train_repair import set_lora
    ok, deadline = c8.budget("Z5")
    _restore(c8)
    if not ok or not c8.has("Z2", "test_units"):
        return {"Z5_skipped": SKIP if not ok else "skipped: no Z2 data"}
    units, harm_tr = c8.load("Z2", "test_units"), c8.load("Z2", "harm")
    model = c8.lora()
    trained = c8.has("Z4", "adapter")
    set_lora(model, c8.load("Z4", "adapter") if trained else c8.lora_init())
    seeds = O.chunk_seeds(c8.run_id, "arms", len(units), c8.gen_bs)
    hseeds = O.chunk_seeds(c8.run_id, "harm", len(harm_tr), c8.gen_bs)
    plan = [("E0", False, False), ("E2", False, True), ("E6", True, False), ("E6h", True, True)]
    out, t0, n_tok = {}, time.time(), 0
    for arm, on, hint in plan:
        if time.time() > deadline:
            m_cut = f"deadline before {arm}"
            break
        seqs = [t["prompt"] + (c8.hint_ids if hint else []) for t in units]
        toks, trunc = gen8_batches(model, seqs, c8.gen_bs, MAX_NEW, c8.eos, c8.pad, seed_fn=lambda c: seeds[c],
                                   budgets=[MAX_NEW - (len(c8.hint_ids) if hint else 0)] * len(seqs), deadline=deadline,
                                   sample=True, adapter_on=on)
        if len(toks) < len(seqs):
            m_cut = f"deadline inside {arm}"
            break
        out[arm] = [(c8.ok(t["pid"], (c8.hint_ids if hint else []) + tk, tr), len(tk) + (len(c8.hint_ids) if hint else 0), tr)
                    for t, tk, tr in zip(units, toks, trunc)]
        n_tok += sum(len(tk) for tk in toks)
    else:
        m_cut = None
    harm = {}
    for name, on in (("adapter", True), ("plain", False)):
        if time.time() > deadline or not harm_tr:
            break
        toks, trunc = gen8_batches(model, [t["prompt"] for t in harm_tr], c8.gen_bs, MAX_NEW, c8.eos, c8.pad,
                                   seed_fn=lambda c: hseeds[c], deadline=deadline, sample=True, adapter_on=on)
        if len(toks) == len(harm_tr):
            harm[name] = [not c8.ok(t["pid"], tk, tr) for t, tk, tr in zip(harm_tr, toks, trunc)]
    rows = []
    for i, t in enumerate(units):
        r = {"pid": t["pid"], "set": t["set"], "group": c8.group(t["pid"])}
        for arm in out:
            r[arm], r[f"tok_{arm}"], r[f"trunc_{arm}"] = out[arm][i]
        rows.append(r)
    c8.save(d, "arms", {"rows": rows, "harm": harm})
    m = {"z5_arms_done": list(out), "z5_units": len(rows), "z5_adapter_trained": trained, "z5_arms_cut": m_cut,
         "z5_gen_tok_per_s": n_tok / max(time.time() - t0, 1e-9), "z5_harm_n": len(harm_tr),
         "z5_harm_adapter": _rate(harm.get("adapter", [])), "z5_harm_plain": _rate(harm.get("plain", []))}
    if all(a in out for a in ARMS):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["set"]].append(r)
            by["pooled"].append(r)
        for s, rs in by.items():
            m[f"z5_{s}"] = arm_stats(rs, [r["group"] for r in rs])
    return m


def z6(c8, d):
    """Greedy pass@1 on all test problems: base (Z2 greedy, cache) vs adapter always on."""
    from train_repair import set_lora
    ok, deadline = c8.budget("Z6")
    _restore(c8)
    if not ok or not c8.has("Z2", "greedy_test"):
        return {"Z6_skipped": SKIP if not ok else "skipped: no Z2 data"}
    base = c8.load("Z2", "greedy_test")
    model = c8.lora()
    trained = c8.has("Z4", "adapter")
    set_lora(model, c8.load("Z4", "adapter") if trained else c8.lora_init())
    t0 = time.time()
    toks, trunc = gen8_batches(model, [t["prompt"] for t in base], c8.gen_bs, MAX_NEW, c8.eos, c8.pad, deadline=deadline,
                               adapter_on=True)
    sec = time.time() - t0
    a6 = [c8.ok(t["pid"], tk, tr) for t, tk, tr in zip(base, toks, trunc)]
    rows = [{"pid": t["pid"], "set": t["set"], "group": c8.group(t["pid"]), "base": t["ok"], "a6": a} for t, a in zip(base, a6)]
    c8.save(d, "a6", rows)
    m = {"z6_n": len(rows), "z6_of": len(base), "z6_adapter_trained": trained,
         "z6_cut": None if len(rows) == len(base) else f"deadline: {len(rows)}/{len(base)} problems",
         "z6_tok_per_s": sum(len(tk) for tk in toks) / max(sec, 1e-9), "z6_a6_trunc": _rate(trunc),
         "z6_a6_mean_tokens": _rate([len(tk) for tk in toks])}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
        by["pooled"].append(r)
    for s, rs in by.items():
        point, ci = O.cluster_bootstrap(rs, lambda xs: O.delta(xs, "a6", "base"), [r["group"] for r in rs])
        m[f"z6_{s}"] = {"n": len(rs), "base": O.recovery(rs, "base"), "a6": O.recovery(rs, "a6"), "delta": point, "delta_ci": ci}
    return m


def z7(c8, d):
    import report
    report.smoke8(c8.run_dir, c8.metrics, c8.cfg)
    return {}


STAGES = [("Z0", z0), ("Z1", z1), ("Z2", z2), ("Z3", z3), ("Z4", z4), ("Z5", z5), ("Z6", z6), ("Z7", z7)]
