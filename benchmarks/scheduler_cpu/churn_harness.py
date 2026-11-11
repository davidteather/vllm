"""Admission-heavy scheduler harness (closed loop, the shape of a serving churn
workload).

Keeps --conc requests outstanding; each has a distinct --prompt-token prompt and --out
output tokens. Finished requests are replaced before the next schedule(), so prefill-
chunk steps admit requests every step at steady state. --out 0 gives a pure-decode
control.

Real (Async)Scheduler + KVCacheManager; fake ModelRunnerOutput. No GPU, no model
weights.

  cd /tmp && PYTHONPATH=<vllm checkout>:<this dir> \
      python churn_harness.py --conc 64 --prompt 1024 --out 16
"""

import argparse
import collections
import gc
import hashlib
import json
import os
import statistics
import time

from sched_harness import (
    allow_spec_on_cpu,
    canon,
    kv_state_hash,
    make_output,
    request_factory,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--conc", type=int, default=64)
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--out", type=int, default=16, help="0 = pure decode control")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--warm", type=int, default=100)
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--async-sched", type=int, default=1)
    p.add_argument("--pc", type=int, default=1)
    p.add_argument("--spec", type=int, default=0)
    p.add_argument("--blocks", type=int, default=40000)
    p.add_argument("--profile", default="")
    p.add_argument("--hash-out", default="")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--patch", default="", help="python module to import first (monkeypatch)"
    )
    return p.parse_args(argv)


def run(args):
    allow_spec_on_cpu()
    if args.patch:
        __import__(args.patch)
    from tests.v1.core.utils import create_scheduler
    from vllm.v1.outputs import DraftTokenIds

    make = request_factory()
    k = args.spec
    sched = create_scheduler(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=max(256, args.conc + 8),
        max_num_batched_tokens=args.budget,
        max_model_len=16384,
        enable_prefix_caching=bool(args.pc),
        num_blocks=args.blocks,
        block_size=16,
        num_speculative_tokens=k or None,
        async_scheduling=bool(args.async_sched),
        skip_tokenizer_init=True,
    )
    max_tokens = args.out if args.out > 0 else 1_000_000
    ctr = [0]
    live = set()

    def add(nreq):
        for _ in range(nreq):
            i = ctr[0]
            ctr[0] += 1
            r = make(f"r{i}", i, args.prompt, max_tokens)
            sched.add_request(r)
            live.add(r.request_id)

    def fake_output(so, prefill_ids):
        req_ids = list(so.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            if rid in prefill_ids:
                sampled.append([])
                continue
            nspec = len(so.scheduled_spec_decode_tokens.get(rid, ()))
            sampled.append([11] * (1 + nspec))
        return make_output(req_ids, sampled)

    add(args.conc)
    is_async = bool(args.async_sched)
    q = collections.deque()
    t_sched, t_upd, t_step, adm, hashes = [], [], [], [], []
    prof = None
    for step in range(args.warm + args.steps):
        timed = step >= args.warm
        if step == args.warm:
            gc.collect()
            gc.freeze()
            if args.profile:
                import cProfile

                prof = cProfile.Profile()
        if args.out > 0 and len(live) < args.conc:
            add(args.conc - len(live))
        if prof and timed:
            prof.enable()
        t0 = time.perf_counter()
        so = sched.schedule()
        t1 = time.perf_counter()
        if prof and timed:
            prof.disable()
        if args.hash_out:
            hashes.append(
                hashlib.sha256(
                    json.dumps(canon(so), sort_keys=True).encode()
                ).hexdigest()
            )
        prefill_ids = {
            rid
            for rid in so.num_scheduled_tokens
            if sched.requests[rid].is_prefill_chunk
        }
        q.appendleft((so, prefill_ids))
        dt_u = 0.0
        if not (is_async and len(q) < 2):
            so_old, pf = q.pop()
            mro = fake_output(so_old, pf)
            if prof and timed:
                prof.enable()
            t2 = time.perf_counter()
            sched.update_from_output(so_old, mro)
            t3 = time.perf_counter()
            if prof and timed:
                prof.disable()
            dt_u = t3 - t2
            if k and not is_async:
                ids = [
                    r
                    for r in so_old.num_scheduled_tokens
                    if r in sched.requests and not sched.requests[r].is_prefill_chunk
                ]
                sched.update_draft_token_ids(
                    DraftTokenIds(ids, [[13] * k for _ in ids])
                )
            for rid in [r for r in live if r not in sched.requests]:
                live.discard(rid)
        if timed:
            t_sched.append(t1 - t0)
            t_upd.append(dt_u)
            t_step.append(t1 - t0 + dt_u)
            adm.append(len(so.scheduled_new_reqs))
    adm_idx = [i for i, a in enumerate(adm) if a]
    res = {
        "conc": args.conc,
        "prompt": args.prompt,
        "out": args.out,
        "async": int(is_async),
        "pc": args.pc,
        "spec": k,
        "steps": args.steps,
        "step_mean_us": statistics.fmean(t_step) * 1e6,
        "sched_mean_us": statistics.fmean(t_sched) * 1e6,
        "sched_med_us": statistics.median(t_sched) * 1e6,
        "upd_mean_us": statistics.fmean(t_upd) * 1e6,
        "upd_med_us": statistics.median(t_upd) * 1e6,
        "adm_steps": len(adm_idx),
        "adm_sched_med_us": statistics.median([t_sched[i] for i in adm_idx]) * 1e6
        if adm_idx
        else 0.0,
        "adm_reqs_per_adm_step": (sum(adm) / len(adm_idx)) if adm_idx else 0,
        "requests_done": ctr[0] - len(live),
    }
    if args.hash_out:
        with open(args.hash_out, "w") as f:
            f.write("\n".join(hashes))
        res["hash_all"] = hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]
        res["kv_hash"] = kv_state_hash(sched)
    if prof:
        prof.dump_stats(args.profile)
    return res


if __name__ == "__main__":
    a = build_args()
    r = run(a)
    print(json.dumps(r) if a.json else r)
