"""Models, prompts, batched generation, prefill features and log-probs (§9.1, §9.3, §9.4).

Generation is HF generate. The KV for all but the last input token is prefilled with the adapter disabled and handed
to generate as past_key_values; generate then runs the last input token and every new token with `adapter_on`. The
same split is used for training (train_repair.py), so the adapter sees identical contexts at training and inference,
and an untrained adapter (B = 0) reproduces the base model exactly. Sampling uses one seed per batch
(torch.manual_seed before generate). Batches are left-padded; positions come from the attention mask.
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


def load_model(repo, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_dir(repo), dtype=dtype,
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


@torch.no_grad()
def decode(model, seqs, max_new, eos, pad, sample=False, seed=0, temperature=0.8, top_p=0.95, adapter_on=False,
           capture=None):
    """Continue each token-id sequence by up to max_new tokens with HF generate (greedy, or temperature/top-p with
    top-k off). Returns (new tokens per row incl. the first EOS, truncated per row, states or None).
    capture = layer indices: states[k][l] is the hidden state [B, d] of the k-th input token generate processed
    (k = 0 the last prefix token, k >= 1 the (k-1)-th new token)."""
    dev = model.device
    ids, mask, pos = _left_pad(seqs, pad, dev)
    with adapter(model, False):
        cache = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1],
                      use_cache=True, logits_to_keep=1).past_key_values
    # Sampling settings go in as kwargs: they override Qwen3's generation_config (do_sample=True, T=0.6, top_k=20).
    kw = (dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if sample
          else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    if sample:
        torch.manual_seed(seed)
    with adapter(model, adapter_on):
        out = model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, max_new_tokens=max_new,
                             eos_token_id=sorted(eos), pad_token_id=pad, return_dict_in_generate=True,
                             output_hidden_states=bool(capture), **kw)
    toks, trunc = [], []
    for row in out.sequences[:, ids.shape[1]:].tolist():
        k = next((i for i, t in enumerate(row) if t in eos), None)
        toks.append(row[:k + 1] if k is not None else row)
        trunc.append(k is None)
    states = ([[step[l][:, -1].float().cpu() for l in capture] for step in out.hidden_states] if capture else None)
    return toks, trunc, states


def decode_batches(model, seqs, bs, max_new, eos, pad, seed_fn=None, budgets=None, **kw):
    """decode() over consecutive chunks of bs rows (callers pass rows sorted by pid, so A2 and A3 get identical
    batches). seed_fn(chunk_index) gives each chunk's sampling seed. budgets (per row) trims each row to its own max."""
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
    """Adapter-off prefill with a fixed batch size; hidden states at the given absolute positions of each sequence:
    list of float32 tensors [n_pos, n_layers, d]. bs=1 means no padding at all."""
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
