#!/usr/bin/env python3
"""Generic config-fan-out job runner (workstream H) for split-inference / training sweeps.

Reads a JSON sweep config: a command template + a list of variants (each = param overrides). Substitutes
params, gives each variant its own run dir + captured log, runs with a concurrency cap, and writes a
manifest. This is the enabler for the static sweeps (A), AE experiments (D), tensor-loss study (E), and
channel sweeps (F) — none of them need a bespoke runner.

Config schema (JSON):
{
  "name": "static_sweep_quant_entropy",
  "cwd": "/abs/path/to/abiodun",            # working dir for each job (optional; default: config's dir/..)
  "env": {"KEY": "VALUE", ...},              # extra env vars (optional)
  "command_template": "python3 ... --quant {quant} --entropy {entropy} --metrics-run-dir {run_dir}",
  "run_dir_template": "metrics_logs/rl_static_sweep/{variant}",   # {variant} + any param names allowed
  "max_parallel": 1,                          # concurrent jobs (loopback+GPU runs: keep 1; offline AE: >1)
  "defaults": {"ckpt": "...", "frames": 300}, # params merged into every variant (variant params win)
  "variants": [
    {"name": "q8_zlib", "params": {"quant": "per_channel_uint8", "entropy": "zlib"}},
    ...
  ]
}
Template placeholders: {variant} (variant name), {run_dir} (resolved abs run dir), and any param key.

Usage:
  python3 sweep_runner.py configs/foo.json --dry-run     # print resolved commands, run nothing
  python3 sweep_runner.py configs/foo.json               # run (concurrency = max_parallel)
  python3 sweep_runner.py configs/foo.json --only q8_zlib,q4_zlib   # subset by variant name
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _resolve(template: str, params: dict) -> str:
    try:
        return template.format(**params)
    except KeyError as e:
        raise SystemExit(f"template references {{{e.args[0]}}} but it is not in defaults/params: {template}")


def build_jobs(cfg: dict, cfg_path: str):
    cwd = cfg.get("cwd") or os.path.abspath(os.path.join(os.path.dirname(cfg_path), os.pardir))
    defaults = cfg.get("defaults", {})
    jobs = []
    for v in cfg["variants"]:
        name = v["name"]
        params = {**defaults, **v.get("params", {}), "variant": name}
        run_dir = os.path.join(cwd, _resolve(cfg["run_dir_template"], params))
        params["run_dir"] = run_dir
        cmd = _resolve(cfg["command_template"], params)
        jobs.append({"name": name, "cmd": cmd, "run_dir": run_dir, "cwd": cwd})
    return jobs, cwd


def run_job(job: dict, env: dict):
    os.makedirs(job["run_dir"], exist_ok=True)
    log = os.path.join(job["run_dir"], "sweep_job.log")
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"# {job['name']}\n# cmd: {job['cmd']}\n# cwd: {job['cwd']}\n\n")
        fh.flush()
        rc = subprocess.run(job["cmd"], shell=True, cwd=job["cwd"], env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    return {"name": job["name"], "returncode": rc, "run_dir": job["run_dir"],
            "duration_s": round(time.time() - t0, 1), "cmd": job["cmd"], "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated variant names to run")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    jobs, cwd = build_jobs(cfg, a.config)
    if a.only:
        want = set(a.only.split(","))
        jobs = [j for j in jobs if j["name"] in want]
    max_par = int(cfg.get("max_parallel", 1))
    env = {**os.environ, **{k: str(v) for k, v in cfg.get("env", {}).items()}}

    print(f"sweep '{cfg.get('name')}': {len(jobs)} job(s), max_parallel={max_par}, cwd={cwd}")
    if a.dry_run:
        for j in jobs:
            print(f"\n[{j['name']}]\n  run_dir: {j['run_dir']}\n  cmd: {j['cmd']}")
        print(f"\n(dry-run: nothing executed)")
        return

    results = []
    with ThreadPoolExecutor(max_workers=max_par) as ex:
        futs = {ex.submit(run_job, j, env): j["name"] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
            print(f"  [{status}] {r['name']} ({r['duration_s']}s) -> {r['run_dir']}")

    manifest = os.path.join(cwd, _resolve(cfg["run_dir_template"], {"variant": "_sweep_manifest"}))
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    mpath = os.path.join(os.path.dirname(manifest), f"{cfg.get('name','sweep')}_manifest.json")
    json.dump({"name": cfg.get("name"), "results": results}, open(mpath, "w"), indent=2)
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"\n{n_ok}/{len(results)} OK. manifest: {mpath}")


if __name__ == "__main__":
    main()
