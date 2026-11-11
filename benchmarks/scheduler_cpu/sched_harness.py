"""Steady-decode scheduler harness: real Scheduler/AsyncScheduler + KVCacheManager, fake
ModelRunnerOutputs. Drives schedule()/update_from_output() the way EngineCore.step
(sync) or step_with_batch_queue (async, queue depth 2) does. No GPU, no model weights.

  cd /tmp && PYTHONPATH=<vllm checkout>:<this dir> \
      python sched_harness.py --n 256 --pc 1 --steps 500
"""

import argparse
import collections
import gc
import hashlib
import json
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256, help="long-running decode requests")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warm", type=int, default=50)
    p.add_argument("--prompt", type=int, default=128)
    p.add_argument("--async-sched", type=int, default=1)
    p.add_argument("--pc", type=int, default=0, help="prefix caching")
    p.add_argument("--spec", type=int, default=0, help="num spec tokens (ngram)")
    p.add_argument(
        "--accept", type=int, default=-1, help="accepted drafts/step; -1=all"
    )
    p.add_argument(
        "--mixed",
        type=int,
        default=0,
        help="add this many --mixed-len prompts every --mixed-every steps",
    )
    p.add_argument("--mixed-every", type=int, default=8)
    p.add_argument("--mixed-len", type=int, default=2048)
    p.add_argument("--mixed-max-tokens", type=int, default=16)
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--blocks", type=int, default=0, help="override num KV blocks")
    p.add_argument("--policy", default="fcfs", choices=["fcfs", "priority"])
    p.add_argument("--profile", default="", help="write cProfile stats here")
    p.add_argument(
        "--hash-out", default="", help="write per-step SchedulerOutput hashes"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--patch", default="", help="python module to import first (monkeypatch)"
    )
    return p.parse_args(argv)


def canon(o):
    """Canonical, process-independent serialization of SchedulerOutput parts."""
    import dataclasses

    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (list, tuple)):
        return [canon(x) for x in o]
    if isinstance(o, (set, frozenset)):
        return sorted(canon(x) for x in o)
    if isinstance(o, dict):
        return sorted(
            ([canon(k), canon(v)] for k, v in o.items()),
            key=lambda kv: json.dumps(kv[0]),
        )
    if dataclasses.is_dataclass(o):
        return {f.name: canon(getattr(o, f.name)) for f in dataclasses.fields(o)}
    return repr(o)


def kv_state_hash(sched):
    """Hash of the final KV-manager state: cached-block counts, per-request block ids,
    prefix-cache keys, free-block count and order."""
    kv = sched.kv_cache_manager
    st = []
    for m in kv.coordinator.single_type_managers:
        st.append(sorted(m.num_cached_block.items()))
        st.append(
            sorted((k, [b.block_id for b in v]) for k, v in m.req_to_blocks.items())
        )
    bp = kv.block_pool
    cm = getattr(bp, "cached_block_hash_to_block", None)
    st.append(sorted(str(x) for x in getattr(cm, "_cache", {})))
    st.append(bp.get_num_free_blocks())
    st.append([b.block_id for b in bp.free_block_queue.get_all_free_blocks()])
    return hashlib.sha256(json.dumps(st, default=str).encode()).hexdigest()[:16]


def allow_spec_on_cpu():
    # Why: the CPU platform rejects spec decode; the scheduler never touches the worker.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    from vllm.platforms import current_platform

    type(current_platform).uses_standalone_cpu_model_runner = classmethod(
        lambda cls: False
    )


def request_factory(block_size=16):
    """Build Requests directly; the tests' create_requests signature drifts."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)

    def make(req_id, seed, num_tokens, max_tokens):
        sp = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
        sp.update_from_generation_config({}, 2)
        toks = [(seed * 7919 + j * 31 + 1) % 150000 for j in range(num_tokens)]
        return Request(
            request_id=req_id,
            prompt_token_ids=toks,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )

    return make


def make_output(req_ids, sampled):
    import dataclasses

    from vllm.v1.outputs import ModelRunnerOutput

    kw = dict(
        req_ids=req_ids,
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    if any(f.name == "req_id_to_index" for f in dataclasses.fields(ModelRunnerOutput)):
        kw["req_id_to_index"] = {r: i for i, r in enumerate(req_ids)}
    return ModelRunnerOutput(**kw)


def run(args):
    allow_spec_on_cpu()
    if args.patch:
        __import__(args.patch)
    from tests.v1.core.utils import create_scheduler
    from vllm.v1.outputs import DraftTokenIds

    n = args.n
    max_seqs = n + (args.mixed * 64 if args.mixed else 0) + 8
    nblk = max(
        20000,
        (n * (args.prompt + (args.steps + args.warm) * (1 + args.spec) + 64)) // 16 * 2
        + args.mixed * 64 * (args.mixed_len // 16 + 4),
    )
    if args.blocks:
        nblk = args.blocks
    sched = create_scheduler(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max(args.budget, n * (1 + args.spec)),
        max_model_len=16384,
        enable_prefix_caching=bool(args.pc),
        num_blocks=nblk,
        block_size=16,
        num_speculative_tokens=args.spec or None,
        async_scheduling=bool(args.async_sched),
        skip_tokenizer_init=True,
        scheduling_policy=args.policy,
    )
    make = request_factory()
    for i in range(n):
        sched.add_request(make(f"d{i}", i, args.prompt, 100000))

    mixed_ctr = [0]

    def add_mixed():
        for i in range(args.mixed):
            j = mixed_ctr[0] + i
            sched.add_request(
                make(f"m{j}", 100000 + j, args.mixed_len, args.mixed_max_tokens)
            )
        mixed_ctr[0] += args.mixed

    accept, k = args.accept, args.spec

    def fake_output(so, prefill_ids):
        req_ids = list(so.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            if rid in prefill_ids:
                sampled.append([])
                continue
            nspec = len(so.scheduled_spec_decode_tokens.get(rid, ()))
            a = nspec if accept < 0 else min(accept, nspec)
            sampled.append([11] * (1 + a))
        return make_output(req_ids, sampled)

    t_sched, t_upd, hashes, reqs_per_step = [], [], [], []
    q = collections.deque()
    is_async = bool(args.async_sched)
    prof = None
    for step in range(args.warm + args.steps):
        timed = step >= args.warm
        if step == args.warm:
            gc.collect()
            gc.freeze()
            if args.profile:
                import cProfile

                prof = cProfile.Profile()
        if args.mixed and step % args.mixed_every == 0 and step >= 5:
            add_mixed()
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
        if timed:
            t_sched.append(t1 - t0)
            reqs_per_step.append(len(so.num_scheduled_tokens))
        q.appendleft((so, prefill_ids))
        if is_async and len(q) < 2:
            continue
        so_old, pf = q.pop()
        mro = fake_output(so_old, pf)
        if prof and timed:
            prof.enable()
        t2 = time.perf_counter()
        sched.update_from_output(so_old, mro)
        t3 = time.perf_counter()
        if prof and timed:
            prof.disable()
        if timed:
            t_upd.append(t3 - t2)
        if k and not is_async:
            ids = [
                r
                for r in so_old.num_scheduled_tokens
                if r in sched.requests and not sched.requests[r].is_prefill_chunk
            ]
            sched.update_draft_token_ids(DraftTokenIds(ids, [[13] * k for _ in ids]))
    res = {
        "n": n,
        "async": int(is_async),
        "pc": args.pc,
        "spec": k,
        "mixed": args.mixed,
        "steps": args.steps,
        "sched_med_us": statistics.median(t_sched) * 1e6,
        "sched_mean_us": statistics.fmean(t_sched) * 1e6,
        "sched_p99_us": sorted(t_sched)[int(len(t_sched) * 0.99)] * 1e6,
        "upd_med_us": statistics.median(t_upd) * 1e6 if t_upd else 0,
        "upd_mean_us": statistics.fmean(t_upd) * 1e6 if t_upd else 0,
        "reqs_per_step": statistics.fmean(reqs_per_step),
        "running_end": len(sched.running),
        "preemptions": sum(r.num_preemptions for r in sched.requests.values()),
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
