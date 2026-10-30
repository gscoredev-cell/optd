"""Models, prompts, batched decoding, prefill features and log-probs (§9.1, §9.3, §9.4).

One decode loop serves every arm: KV for all but the last input token is built with the adapter disabled; the last
input token and every new token run with `adapter_on`. The same split is used for training (train_repair.py), so the
adapter sees identical contexts at training and inference, and an untrained adapter (B = 0) reproduces the base model
exactly. Batches are left-padded; positions come from the attention mask.
"""
import contextlib
import pathlib

import torch

from core import SYSTEM


def model_dir(repo):
    return pathlib.Path("/workspace/models") / repo.split("/")[-1]


def load_tokenizer(repo):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir(repo))
    tok.padding_side = "left"
    return tok


def load_model(repo):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_dir(repo), dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda").eval()


def eos_ids(model):
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


def prompt_ids(tok, question):
    return list(tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                                        tokenize=True, add_generation_prompt=True, enable_thinking=False))


def adapter(model, on):
    """Context with the LoRA adapter enabled (on) or disabled; a plain model has no adapter."""
    return contextlib.nullcontext() if on or not hasattr(model, "disable_adapter") else model.disable_adapter()


def _left_pad(seqs, pad, device):
    width = max(len(s) for s in seqs)
    ids = torch.tensor([[pad] * (width - len(s)) + list(s) for s in seqs], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
    return ids, mask, (mask.cumsum(-1) - 1).clamp(min=0)


def _top_p_sample(logits, temperature, top_p, gen):
    probs = torch.softmax(logits / temperature, -1)
    sp, si = probs.sort(-1, descending=True)
    sp = sp * ((sp.cumsum(-1) - sp) < top_p)   # keep the smallest prefix reaching top_p (always the top token)
    return si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)).squeeze(-1)


@torch.no_grad()
def decode(model, seqs, max_new, eos, pad, sample=False, seed=0, temperature=0.8, top_p=0.95, adapter_on=False,
           capture=None):
    """Continue each token-id sequence by up to max_new tokens (greedy or top-p sampling with one generator per
    batch). Returns (new_tokens per row incl. the first EOS, truncated per row, captured states or None).
    capture = layer indices: states[k][l] is the hidden state of the k-th input token fed during decoding
    (k = 0 is the last prefix token, k >= 1 the (k-1)-th new token), shape [B, d]."""
    dev = model.device
    ids, mask, pos = _left_pad(seqs, pad, dev)
    with adapter(model, False):
        cache = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1],
                      use_cache=True, logits_to_keep=1).past_key_values
    gen = torch.Generator(device=dev).manual_seed(seed) if sample else None
    eos_t = torch.tensor(sorted(eos), device=dev)
    cur, cur_pos = ids[:, -1:], pos[:, -1:]
    done = torch.zeros(len(seqs), dtype=torch.bool, device=dev)
    out, states = [], [] if capture else None
    with adapter(model, adapter_on):
        for _ in range(max_new):
            r = model(input_ids=cur, attention_mask=mask, position_ids=cur_pos, past_key_values=cache,
                      use_cache=True, output_hidden_states=bool(capture))
            cache = r.past_key_values
            if capture:
                states.append([r.hidden_states[l][:, -1].float().cpu() for l in capture])
            logits = r.logits[:, -1].float()
            nxt = _top_p_sample(logits, temperature, top_p, gen) if sample else logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            out.append(nxt)
            done |= torch.isin(nxt, eos_t)
            if done.all():
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            cur, cur_pos = nxt[:, None], cur_pos + 1
    rows = torch.stack(out, 1).tolist()
    toks, trunc = [], []
    for row in rows:
        k = next((i for i, t in enumerate(row) if t in eos), None)
        toks.append(row[:k + 1] if k is not None else row)
        trunc.append(k is None)
    return toks, trunc, states


def decode_batches(model, seqs, bs, max_new, eos, pad, seed_fn=None, budgets=None, **kw):
    """decode() over consecutive chunks of bs rows (callers pass rows sorted by pid). seed_fn(chunk_index) gives each
    chunk's sampling seed. budgets (per row) trims each row's output to its own max and recomputes truncation."""
    toks, trunc = [], []
    for c, i in enumerate(range(0, len(seqs), bs)):
        chunk = seqs[i:i + bs]
        b = [max(1, x) for x in budgets[i:i + bs]] if budgets else [max_new] * len(chunk)
        t, _, _ = decode(model, chunk, max(b), eos, pad, seed=seed_fn(c) if seed_fn else 0, **kw)
        for row, lim in zip(t, b):
            cut = row[:lim]
            toks.append(cut)
            trunc.append(not (cut and cut[-1] in eos))
    return toks, trunc


@torch.no_grad()
def prefill_states(model, seqs, positions, layers, bs):
    """Adapter-off prefill; hidden states at the given absolute positions of each sequence:
    list of float32 tensors [n_pos, n_layers, d]."""
    out = []
    for i in range(0, len(seqs), bs):
        chunk, pchunk = seqs[i:i + bs], positions[i:i + bs]
        ids, mask, pos = _left_pad(chunk, 0, model.device)
        with adapter(model, False):
            hs = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
                       logits_to_keep=1).hidden_states
        width = ids.shape[1]
        for r, (s, ps) in enumerate(zip(chunk, pchunk)):
            idx = torch.tensor([width - len(s) + p for p in ps], device=model.device, dtype=torch.long)
            out.append(torch.stack([hs[l][r, idx].float().cpu() for l in layers], 1) if ps
                       else torch.zeros(0, len(layers), hs[0].shape[-1]))
        del hs
    return out


@torch.no_grad()
def token_logprobs(model, prefixes, conts, bs, adapter_on=False):
    """log p(cont_t | prefix, cont_<t) for each pair, full prefill (adapter off unless adapter_on). [list of float32]"""
    out = []
    for i in range(0, len(prefixes), bs):
        seqs = [p + c for p, c in zip(prefixes[i:i + bs], conts[i:i + bs])]
        ids, mask, pos = _left_pad(seqs, 0, model.device)
        with adapter(model, adapter_on):
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
        width = ids.shape[1]
        for r, (p, c) in enumerate(zip(prefixes[i:i + bs], conts[i:i + bs])):
            start = width - len(c) - 1   # logits at position t predict token t+1
            lp = logits[r, start:start + len(c)].float().log_softmax(-1)
            out.append(lp.gather(-1, torch.tensor(c, dtype=torch.long, device=lp.device)[:, None]).squeeze(-1).cpu())
        del logits
    return out
