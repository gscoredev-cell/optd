"""One long-lived process per phase (OPTION_D_SPEC.md §13): ENVCHECK, SMOKE (and WATCHDOGTEST for the deadline test).

Each stage writes runs/<RUN_ID>/stage_<name>/ and is skipped on restart once its .done marker exists.
Never prints or stores question text, secrets or environment dumps.
"""
import argparse
import json
import pathlib
import sys
import time

import torch

from core import write_status

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMPTY_THINK = "<think>\n\n</think>\n\n"


def gsm_rows(cfg, config):
    from huggingface_hub import snapshot_download
    import data
    return data.load_gsm(snapshot_download(cfg["dataset"], repo_type="dataset", local_files_only=True))[config]


def chat_text(tok, question):
    from core import SYSTEM
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def stage_env(d):
    import peft
    import transformers
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    rel = ((c.float() - ref).norm() / ref.norm()).item()
    assert torch.isfinite(c).all() and rel < 1e-2, f"bf16 matmul check failed: rel err {rel:.3g}"
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft.__version__, "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()), "bf16_matmul_rel_err": round(rel, 5),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


def stage_templates(d, cfg):
    from generate import load_tokenizer, prompt_ids
    ts, tt = load_tokenizer(cfg["student"]), load_tokenizer(cfg["teacher"])
    questions = [r["question"] for r in gsm_rows(cfg, "main")[:20]]
    same_ids = sum(prompt_ids(ts, q) == prompt_ids(tt, q) for q in questions)
    think = sum(chat_text(ts, q).endswith(EMPTY_THINK) and chat_text(ts, q) == chat_text(tt, q) for q in questions)
    assert same_ids == 20, f"student/teacher token ids differ on {20 - same_ids}/20 prompts"
    assert think == 20, f"empty think block missing or different on {20 - think}/20 prompts"
    return {"template_identical_ids": same_ids, "template_empty_think": think, "template_n": 20}


def stage_gen(d, cfg, tiers, tier):
    from generate import load_model, load_tokenizer
    tok = load_tokenizer(cfg["student"])
    bs = tiers[tier]["gen_batch"]
    enc = tok([chat_text(tok, r["question"]) for r in gsm_rows(cfg, "p2")[:bs]], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    model = load_model(cfg["student"])
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    sec = time.time() - t0
    plen = enc.input_ids.shape[1]
    lengths = [next((i + 1 for i, t in enumerate(row) if t in eos), len(row)) for row in out[:, plen:].tolist()]
    truncated = sum(not any(t in eos for t in row) for row in out[:, plen:].tolist())
    torch.save({"input_ids": out.cpu(), "prompt_mask": enc.attention_mask.cpu(), "prompt_len": plen,
                "lengths": lengths}, d / "gen.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return {"gen_batch": bs, "gen_tokens": sum(lengths), "gen_sec": round(sec, 2),
            "gen_tok_per_s": round(sum(lengths) / sec, 1), "gen_truncated": truncated,
            "gen_peak_vram_gb": round(peak, 1)}


def stage_teacher_prefill(d, cfg, tiers, tier, gen_dir):
    from generate import load_model
    g = torch.load(gen_dir / "gen.pt")
    bs = tiers[tier]["teacher_prefill_batch"]
    plen, lengths = g["prompt_len"], g["lengths"][:bs]
    width = plen + max(lengths)
    ids = g["input_ids"][:bs, :width].cuda()
    gen_mask = (torch.arange(width - plen)[None, :] < torch.tensor(lengths)[:, None]).long()
    mask = torch.cat([g["prompt_mask"][:bs], gen_mask], dim=1).cuda()
    pos = (mask.cumsum(-1) - 1).clamp(min=0)   # left padding: positions as during generation
    model = load_model(cfg["teacher"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
    torch.cuda.synchronize()
    sec = time.time() - t0
    with torch.no_grad():   # sanity: mean teacher log-prob of the student's tokens is finite
        lp = [logits[i, plen - 1:plen - 1 + n].float().log_softmax(-1)
              .gather(-1, ids[i, plen:plen + n, None]).mean().item() for i, n in enumerate(lengths)]
    peak = torch.cuda.max_memory_allocated() / 2**30
    tokens = int(mask.sum())
    del logits, model
    torch.cuda.empty_cache()
    assert all(map(lambda x: x == x and abs(x) < 1e4, lp)), "non-finite teacher log-probs"
    return {"prefill_batch": bs, "prefill_tokens": tokens, "prefill_sec": round(sec, 3),
            "prefill_tok_per_s": round(tokens / sec, 1), "teacher_mean_logp": round(sum(lp) / len(lp), 3),
            "prefill_peak_vram_gb": round(peak, 1)}


def run_stage(run_dir, name, fn, metrics):
    d = run_dir / f"stage_{name}"
    if (d / ".done").exists():
        print(f"== stage {name}: skipped (.done)")
        metrics.update(json.loads((d / "metrics.json").read_text()))
        return
    d.mkdir(parents=True, exist_ok=True)
    write_status(run_dir, name)
    print(f"== stage {name}")
    t0 = time.time()
    m = fn(d)
    m[f"{name}_min"] = round((time.time() - t0) / 60, 2)
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    (d / ".done").touch()
    metrics.update(m)
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, (dict, list))}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", type=int, required=True)
    a = ap.parse_args()
    run_dir = pathlib.Path("/workspace/runs") / a.run_id
    if a.phase == "WATCHDOGTEST":   # only the in-pod deadline watchdog may end this pod
        write_status(run_dir, "watchdogtest_sleep")
        print("== WATCHDOGTEST: sleeping 30 min")
        time.sleep(1800)
        return
    import yaml
    import report
    cfg = yaml.safe_load((ROOT / "config/run_defaults.yaml").read_text())
    tiers = yaml.safe_load((ROOT / "config/gpu_tiers.yaml").read_text())["tiers"]
    try:
        m = {"pipeline_start": time.time()}
        run_stage(run_dir, "env", stage_env, m)
        run_stage(run_dir, "templates", lambda d: stage_templates(d, cfg), m)
        if a.phase == "ENVCHECK":
            run_stage(run_dir, "gen", lambda d: stage_gen(d, cfg, tiers, a.tier), m)
            run_stage(run_dir, "teacher_prefill",
                      lambda d: stage_teacher_prefill(d, cfg, tiers, a.tier, run_dir / "stage_gen"), m)
            write_status(run_dir, "report")
            report.envcheck(run_dir, m, cfg)
        elif a.phase == "SMOKE":
            import smoke
            ctx = smoke.Ctx(cfg, tiers, a.tier, a.run_id, run_dir)
            for name, fn in smoke.STAGES:
                run_stage(run_dir, name, lambda d, fn=fn: fn(ctx, d), m)
            write_status(run_dir, "report")
            report.smoke(run_dir, m, cfg)
        elif a.phase == "SMOKE3":
            import smoke3
            c3 = smoke3.Ctx3(cfg, tiers, a.tier, a.run_id, run_dir)
            for name, fn in smoke3.STAGES:   # diagnostic parts are independent: a failing one must not cost the report
                try:
                    run_stage(run_dir, name, lambda d, fn=fn: fn(c3, d), m)
                except Exception as e:
                    m[f"{name}_skipped"] = f"error: {type(e).__name__}: {(str(e).splitlines() or [''])[0][:200]}"
                    print(f"== stage {name} failed: {m[f'{name}_skipped']}")
                    smoke3.smoke.free()
            write_status(run_dir, "report")
            report.smoke3(run_dir, m, cfg)
        else:
            raise NotImplementedError(f"phase {a.phase} is not implemented (FULL needs human approval of SMOKE)")
    except BaseException as e:
        (run_dir / ".err").write_text(f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"[:300])
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
