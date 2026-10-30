"""One long-lived process per phase (OPTION_D_SPEC.md §13). Implemented so far: ENVCHECK.

Each stage writes runs/<RUN_ID>/stage_<name>/ and is skipped on restart once its .done marker exists.
Never prints or stores question text, secrets or environment dumps.
"""
import argparse
import gc
import json
import os
import pathlib
import time

import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config/run_defaults.yaml").read_text())
TIERS = yaml.safe_load((ROOT / "config/gpu_tiers.yaml").read_text())["tiers"]
SYSTEM = ("Solve the problem step by step. Write each step on its own line. "
          "End with exactly one line: The final answer is <number>")
MAX_NEW = 400
EMPTY_THINK = "<think>\n\n</think>\n\n"


def write_status(run_dir, step):
    (run_dir / ".step").write_text(step)
    tmp = run_dir / f".status.py.{os.getpid()}"
    tmp.write_text(json.dumps({"run_id": run_dir.name, "state": "RUNNING", "step": step, "msg": "",
                               "ts": int(time.time())}, separators=(",", ":")))
    os.replace(tmp, run_dir / "status.json")


def model_dir(repo):
    return pathlib.Path("/workspace/models") / repo.split("/")[-1]


def gsm(config):
    from huggingface_hub import snapshot_download
    root = pathlib.Path(snapshot_download(CFG["dataset"], repo_type="dataset", local_files_only=True))
    return [json.loads(l) for l in (root / config / "test.jsonl").read_text().splitlines() if l.strip()]


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def chat(tok, question, tokenize=False):
    return tok.apply_chat_template(messages(question), tokenize=tokenize, add_generation_prompt=True,
                                   enable_thinking=False)


def tokenizer(repo):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir(repo))
    tok.padding_side = "left"
    return tok


def load(repo):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_dir(repo), dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda").eval()


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


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


def stage_templates(d):
    ts, tt = tokenizer(CFG["student"]), tokenizer(CFG["teacher"])
    questions = [r["question"] for r in gsm("main")[:20]]
    same_ids = sum(chat(ts, q, True) == chat(tt, q, True) for q in questions)
    think = sum(chat(ts, q).endswith(EMPTY_THINK) and chat(ts, q) == chat(tt, q) for q in questions)
    assert same_ids == 20, f"student/teacher token ids differ on {20 - same_ids}/20 prompts"
    assert think == 20, f"empty think block missing or different on {20 - think}/20 prompts"
    return {"template_identical_ids": same_ids, "template_empty_think": think, "template_n": 20}


def stage_gen(d, tier):
    tok = tokenizer(CFG["student"])
    bs = TIERS[tier]["gen_batch"]
    enc = tok([chat(tok, r["question"]) for r in gsm("p2")[:bs]], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    model = load(CFG["student"])
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    sec = time.time() - t0
    plen = enc.input_ids.shape[1]
    lengths = [next((i + 1 for i, t in enumerate(row) if t in eos), len(row)) for row in out[:, plen:].tolist()]
    truncated = sum(not any(t in eos for t in row) for row in out[:, plen:].tolist())
    torch.save({"input_ids": out.cpu(), "prompt_mask": enc.attention_mask.cpu(), "prompt_len": plen,
                "lengths": lengths}, d / "gen.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30
    unload(model)
    return {"gen_batch": bs, "gen_tokens": sum(lengths), "gen_sec": round(sec, 2),
            "gen_tok_per_s": round(sum(lengths) / sec, 1), "gen_truncated": truncated,
            "gen_peak_vram_gb": round(peak, 1)}


def stage_teacher_prefill(d, tier, gen_dir):
    g = torch.load(gen_dir / "gen.pt")
    bs = TIERS[tier]["teacher_prefill_batch"]
    plen, lengths = g["prompt_len"], g["lengths"][:bs]
    width = plen + max(lengths)
    ids = g["input_ids"][:bs, :width].cuda()
    gen_mask = (torch.arange(width - plen)[None, :] < torch.tensor(lengths)[:, None]).long()
    mask = torch.cat([g["prompt_mask"][:bs], gen_mask], dim=1).cuda()
    pos = (mask.cumsum(-1) - 1).clamp(min=0)   # left padding: positions as during generation
    model = load(CFG["teacher"])
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
    del logits
    unload(model)
    assert all(map(lambda x: x == x and abs(x) < 1e4, lp)), "non-finite teacher log-probs"
    return {"prefill_batch": bs, "prefill_tokens": tokens, "prefill_sec": round(sec, 3),
            "prefill_tok_per_s": round(tokens / sec, 1), "teacher_mean_logp": round(sum(lp) / len(lp), 3),
            "prefill_peak_vram_gb": round(peak, 1)}


def write_report(run_dir, m):
    env = json.loads((run_dir / "env.json").read_text())
    dl = [json.loads(l) for l in (run_dir / "downloads.jsonl").read_text().splitlines() if l.strip()]
    hits = sum(x["cache"] == "hit" for x in dl)
    got = [x for x in dl if x["cache"] == "download"]
    mbps = f"{sum(x['mb'] for x in got) / max(sum(x['sec'] for x in got), 1e-3):.1f}" if got else "n/a (all cache hits)"
    boot = int(os.environ["BOOT_EPOCH"])
    elapsed = (time.time() - boot) / 60
    price = float(env["price_per_hr"])
    m.update({"elapsed_min": round(elapsed, 2), "cost_usd": round(price * elapsed / 60, 3), "cache_hits": hits,
              "downloads": len(got), "bootstrap_min": round((m["pipeline_start"] - boot) / 60, 2)})
    cap = ".".join(map(str, m["capability"]))
    lines = [
        f"RUN: {env['run_id']} | {env['phase']} | {env['git_sha'][:7]} | {env['pod_id']} | {env['gpu']} | "
        f"{m['vram_gb']} GB | {CFG['datacenter']} | ${price:.2f}/h | {elapsed:.1f} min | ${price * elapsed / 60:.2f}",
        f"ENV: {CFG['image']} | torch {m['torch']} (cuda {m['cuda']}) | transformers {m['transformers']} | "
        f"peft {m['peft']} | {CFG['student'].split('/')[-1]} / {CFG['teacher'].split('/')[-1]} | "
        f"cache hits {hits} / downloads {len(got)} | download MB/s {mbps}",
        f"DEVICE: capability {cap} | bf16 matmul ok (rel err {m['bf16_matmul_rel_err']}) | tier {env['tier']} | "
        f"driver {env['driver']}",
        f"TEMPLATES: identical token ids {m['template_identical_ids']}/{m['template_n']} | "
        f"empty think block {m['template_empty_think']}/{m['template_n']}",
        f"TIMING (min): bootstrap {m['bootstrap_min']} | env {m['env_min']} | templates {m['templates_min']} | "
        f"gen {m['gen_min']} | teacher_prefill {m['teacher_prefill_min']} | total {elapsed:.2f}",
        f"THROUGHPUT: gen tok/s {m['gen_tok_per_s']} (batch {m['gen_batch']}, {m['gen_tokens']} tokens, "
        f"truncated {m['gen_truncated']}/{m['gen_batch']}, peak {m['gen_peak_vram_gb']} GB) | "
        f"teacher prefill tok/s {m['prefill_tok_per_s']} (batch {m['prefill_batch']}, {m['prefill_tokens']} tokens, "
        f"peak {m['prefill_peak_vram_gb']} GB, mean logp {m['teacher_mean_logp']})",
        "DOWNLOADS: " + " ; ".join(f"{x['repo']} cache hit" if x["cache"] == "hit"
                                   else f"{x['repo']} {x['mb']} MB {x['mb_per_s']} MB/s" for x in dl),
        f"BUDGET: max_minutes {env['max_minutes']} x ${price:.2f}/h = ${price * int(env['max_minutes']) / 60:.2f} "
        f"<= ${CFG['phase_budget_usd'][env['phase']]}",
        "BLOCKERS: none",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    print("\n".join(lines))


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
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    (d / ".done").touch()
    metrics.update(m)
    print(json.dumps(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", type=int, required=True)
    a = ap.parse_args()
    run_dir = pathlib.Path("/workspace/runs") / a.run_id
    try:
        if a.phase != "ENVCHECK":
            raise NotImplementedError(f"phase {a.phase} is not implemented yet")
        m = {"pipeline_start": time.time()}
        run_stage(run_dir, "env", stage_env, m)
        run_stage(run_dir, "templates", stage_templates, m)
        run_stage(run_dir, "gen", lambda d: stage_gen(d, a.tier), m)
        run_stage(run_dir, "teacher_prefill",
                  lambda d: stage_teacher_prefill(d, a.tier, run_dir / "stage_gen"), m)
        write_status(run_dir, "report")
        write_report(run_dir, m)
    except BaseException as e:
        (run_dir / ".err").write_text(f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"[:300])
        raise


if __name__ == "__main__":
    main()
