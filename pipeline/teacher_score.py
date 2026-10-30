"""S6: per-token log-probs of each continuation under the teacher and under the student that sampled it (§9.4)."""
from generate import token_logprobs


def score(model, conts, bs, adapter_on=False):
    # ponytail: each continuation re-prefills its shared prefix (k x the prefix work); prefill-once-per-problem
    # with cache reuse if S6 time matters at FULL scale.
    return token_logprobs(model, [c["prefix"] for c in conts], [c["cont"] for c in conts], bs, adapter_on)
