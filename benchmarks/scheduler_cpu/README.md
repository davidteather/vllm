# CPU-only scheduler harness for vLLM v1

Drives the real `Scheduler`/`AsyncScheduler` and `KVCacheManager` with fake
`ModelRunnerOutput`s, the way `EngineCore.step_with_batch_queue` does, on a laptop: no GPU, no
weights, no engine process. Used for the numbers posted on vllm-project/vllm#63934 and in
#66176.

| file | what |
|---|---|
| `sched_harness.py` | steady decode: `--n` long-running requests, optional ngram spec (`--spec`), optional periodic long-prompt admissions (`--mixed`), sync or async scheduling |
| `churn_harness.py` | closed-loop churn: `--conc` requests of `--prompt` tokens × `--out` output tokens, finished requests replaced each step (admissions every step); `--out 0` is a pure-decode control |
| `ab.py` | two commits → two worktrees → identity check over 7 configs, then alternated timing rounds in fresh processes, markdown table |

Both harnesses report median/mean µs per step for `schedule()` and `update_from_output()`,
and with `--hash-out` a SHA of every `SchedulerOutput` (canonicalised) plus a hash of the final
KV-manager state (cached-block counts, per-request block ids, prefix-cache keys, free-block
count and order). Two commits are behaviourally identical on a config when both hashes match.

## Requirements

A vLLM checkout with an editable install (`VLLM_USE_PRECOMPILED=1 uv pip install -e .` or a CPU
build) whose `tests/` directory is importable; the harness uses `tests.v1.core.utils.create_scheduler`
and `create_requests`. Tested with Python 3.12 on macOS arm64 (CPU build) and Linux.

## One side

```bash
cd /tmp   # not the checkout: cwd must not shadow the installed package
PYTHONPATH=/path/to/vllm:/path/to/this/dir /path/to/vllm/.venv/bin/python \
    /path/to/this/dir/sched_harness.py --n 256 --pc 1 --steps 1500 --warm 100 --json
PYTHONPATH=... python churn_harness.py --conc 64 --prompt 1024 --out 16 --steps 600 --json
```

## A/B two commits with identity check

```bash
python ab.py --repo /path/to/vllm --python /path/to/vllm/.venv/bin/python \
    --base <base sha> --pr <pr head sha> --rounds 5
```

`ab.py` makes `git worktree`s for both commits under `--scratch` (default `$TMPDIR/vllm-sched-ab`)
and symlinks the repo's ignored build artifacts (`git ls-files --others --ignored vllm`) into each,
so both sides share binaries and differ only in Python. Rounds alternate base/pr; report the median
and the spread. Keep the machine quiet (the load average is printed per round); re-run if spread
exceeds ~10% of the value.

## Reading the numbers

- Harness µs are CPU cost of the scheduler step, not serving latency. On an H100 with async
  scheduling the decode step's host work is hidden behind the GPU unless the host is longer than
  the forward pass, so a scheduler saving shows up as engine CPU and (under admissions) TTFT, not
  tok/s. Say which it is.
- `sched_harness` `--n 64/256` is the steady-decode floor; `churn_harness --out 16` is where
  `allocate_slots`/`cache_blocks` admission costs dominate.
- Identity across the seven configs covers: async and sync scheduling, spec decode with accepted
  drafts, mixed long-prompt admissions with chunked prefill, prefix caching, and churn with
  finishing requests and block eviction.
