"""S7 / E-S5: LoRA repair adapter (OPTION_D_SPEC §9.4, OPTION_E_SPEC §4). r = 16 on all linear layers, B = 0 at init;
the prefix KV (all but the last prefix token) is built with the adapter disabled and without grad, the last prefix
token and the continuation run with the adapter enabled, matching generate.decode(). Loss rkl (default):
A_t = clip(-(logp_S_old - logp_T), -5, 5), loss = -mean_t A_t * logp_theta(y_t); rft: plain SFT on the given (correct)
continuations. AdamW lr 1e-4, cosine with 3% warmup, effective batch 16, one epoch; checkpoints every ckpt_every
optimizer steps; resume=True continues from the latest checkpoint.

Early stopping (Option E): eval_fn(model) -> dev score every eval_every steps (and at step 0 and the end); the best
state is kept and training stops after `patience` evaluations without improvement. `early` (a dict) receives the curve,
best step/score/state and whether the stop fired. Sloppy-cut mixing: sloppy_mix() adds b*-1 copies of a share of units.
torch is imported lazily so the pure helpers are testable without it."""
import math
import pathlib
import random

from core import seed_for


def sloppy_mix(units, frac=0.25, seed=0):
    """Add a copy with cut b-1 for `frac` of the units with b >= 1 (seeded choice). Each copy carries sloppy=True."""
    cands = [i for i, u in enumerate(units) if u["b"] >= 1]
    pick = set(random.Random(seed).sample(cands, round(frac * len(cands))))
    return list(units) + [{**units[i], "b": units[i]["b"] - 1, "sloppy": True} for i in sorted(pick)]


def wrap_lora(model, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)   # lora_A init
    return get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"))


def lora_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().clone().cpu() for k, v in get_peft_model_state_dict(model).items()}


def set_lora(model, state):
    from peft import set_peft_model_state_dict
    res = set_peft_model_state_dict(model, state)   # strict=False inside peft: a key mismatch would load nothing
    bad = getattr(res, "unexpected_keys", [])
    assert not bad, f"adapter state did not match the model: {len(bad)} unexpected keys"


def _loss(model, ex, loss_kind, n_tokens):
    import torch
    from generate import _left_pad, adapter
    dev = model.device
    prefixes, conts = [e["prefix"] for e in ex], [e["cont"] for e in ex]
    ids_p, mask_p, pos_p = _left_pad(prefixes, 0, dev)
    with torch.no_grad(), adapter(model, False):
        cache = model(input_ids=ids_p[:, :-1], attention_mask=mask_p[:, :-1], position_ids=pos_p[:, :-1],
                      use_cache=True, logits_to_keep=1).past_key_values
    T = max(len(c) for c in conts)
    inp = torch.tensor([[p[-1]] + c[:-1] + [0] * (T - len(c)) for p, c in zip(prefixes, conts)], device=dev)
    tgt = torch.tensor([c + [0] * (T - len(c)) for c in conts], device=dev)
    valid = torch.tensor([[1] * len(c) + [0] * (T - len(c)) for c in conts], device=dev)
    pos = pos_p[:, -1:] + torch.arange(T, device=dev)[None]
    with adapter(model, True):
        logits = model(input_ids=inp, attention_mask=torch.cat([mask_p[:, :-1], valid], 1), position_ids=pos,
                       past_key_values=cache).logits.float()
    lp = logits.log_softmax(-1).gather(-1, tgt[..., None]).squeeze(-1)
    if loss_kind == "rkl":
        pad = lambda key: torch.stack([torch.nn.functional.pad(e[key], (0, T - len(e[key]))) for e in ex]).to(dev)
        w = (-(pad("lp_s") - pad("lp_t"))).clamp(-5, 5)
    else:
        w = torch.ones_like(lp)
    return -(w * lp * valid).sum() / n_tokens


def _evaluate(model, eval_fn, step, early):
    model.eval()
    score = eval_fn(model)
    model.train()
    early.setdefault("curve", []).append([step, score])
    if early.get("best_score") is None or score > early["best_score"]:
        early.update({"best_score": score, "best_step": step, "best_state": lora_state(model), "bad": 0})
    else:
        early["bad"] = early.get("bad", 0) + 1
    return early["bad"] >= early.get("patience", 3)


def train(model, data, ckpt_dir, run_id, loss_kind="rkl", resume=False, stop_at=None, ckpt_every=10, micro=4,
          eff=16, lr=1e-4, eval_fn=None, eval_every=10, patience=3, early=None):
    """One epoch over data (dicts prefix, cont, lp_t, lp_s). Returns ({step: loss}, total optimizer steps)."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(data)))
    random.Random(seed_for(run_id, "train", 0)).shuffle(order)
    steps = math.ceil(len(order) / eff)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.03 * steps), steps)
    start, losses = 0, {}
    early = early if early is not None else {}
    early["patience"] = patience
    if resume:
        st = torch.load(max(ckpt_dir.glob("step_*.pt")), weights_only=False)
        set_lora(model, st["lora"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, losses = st["step"], dict(st["losses"])
        early.update(st.get("early", {}))
    model.train()
    if eval_fn and not resume:
        _evaluate(model, eval_fn, 0, early)
    stopped = False
    step = start
    for step in range(start, steps):
        if stop_at is not None and step >= stop_at:
            break
        idx = order[step * eff:(step + 1) * eff]
        n_tokens = sum(len(data[i]["cont"]) for i in idx)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for j in range(0, len(idx), micro):
            loss = _loss(model, [data[i] for i in idx[j:j + micro]], loss_kind, n_tokens)
            loss.backward()
            total += loss.item()
        opt.step()
        sched.step()
        losses[step + 1] = total
        if eval_fn and (step + 1) % eval_every == 0 and _evaluate(model, eval_fn, step + 1, early):
            stopped = True
        if (step + 1) % ckpt_every == 0:
            torch.save({"step": step + 1, "lora": lora_state(model), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "losses": losses, "early": dict(early)},
                       ckpt_dir / f"step_{step + 1:05d}.pt")
        if stopped:
            break
    if eval_fn and not stopped and (stop_at is None or stop_at >= steps) and (step + 1) % eval_every:
        _evaluate(model, eval_fn, step + 1, early)   # final evaluation when the epoch ended off the grid
    early["stopped"] = stopped
    early["last_step"] = step + 1 if steps else 0
    model.eval()
    return losses, steps
