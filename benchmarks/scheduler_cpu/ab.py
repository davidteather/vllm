"""A/B two commits of vLLM on the CPU-only scheduler harnesses, with an output-identity
check.

For each side a detached git worktree is created under --scratch and the editable
build's ignored artifacts (compiled extensions, generated version file) are symlinked
in from --repo, so both sides import the same binaries and differ only in Python
source. Timing rounds alternate sides (ABBA) in fresh processes; the identity check
hashes every SchedulerOutput and the final KV-manager state across several scheduler
configurations.

  python ab.py --repo ~/vllm --python ~/vllm/.venv/bin/python \
      --base <sha> --pr <sha> --rounds 5
"""

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

TIMING = {
    "decode": [
        (
            "sched_harness.py",
            ["--n", "64", "--pc", "1", "--steps", "1500", "--warm", "100"],
        ),
        (
            "sched_harness.py",
            ["--n", "256", "--pc", "1", "--steps", "1500", "--warm", "100"],
        ),
    ],
    "churn": [
        (
            "churn_harness.py",
            ["--conc", "64", "--prompt", "1024", "--out", "16", "--steps", "600"],
        ),
        (
            "churn_harness.py",
            ["--conc", "64", "--prompt", "1024", "--out", "0", "--steps", "600"],
        ),
    ],
}
IDENTITY = [
    ("sched_harness.py", ["--n", "64", "--pc", "1", "--steps", "300"]),
    ("sched_harness.py", ["--n", "64", "--pc", "1", "--steps", "300", "--spec", "3"]),
    (
        "sched_harness.py",
        [
            "--n",
            "64",
            "--pc",
            "1",
            "--steps",
            "300",
            "--mixed",
            "4",
            "--mixed-max-tokens",
            "37",
            "--mixed-len",
            "1003",
        ],
    ),
    (
        "sched_harness.py",
        ["--n", "64", "--pc", "1", "--steps", "300", "--async-sched", "0"],
    ),
    ("sched_harness.py", ["--n", "256", "--pc", "1", "--steps", "200"]),
    (
        "churn_harness.py",
        ["--conc", "64", "--prompt", "1024", "--out", "16", "--steps", "300"],
    ),
    (
        "churn_harness.py",
        [
            "--conc",
            "64",
            "--prompt",
            "1024",
            "--out",
            "16",
            "--steps",
            "300",
            "--spec",
            "3",
        ],
    ),
]


def sh(*args, cwd=None):
    r = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if r.returncode:
        sys.exit(f"{' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def worktree(repo, scratch, sha):
    full = sh("git", "rev-parse", sha, cwd=repo).strip()
    wt = scratch / f"wt-{full[:10]}"
    if not (wt / ".git").exists():
        sh("git", "worktree", "add", "--detach", str(wt), full, cwd=repo)
    for rel in sh(
        "git",
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "vllm",
        cwd=repo,
    ).splitlines():
        if "__pycache__" in rel or rel.endswith("/"):
            continue
        dst = wt / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(repo / rel)
    return wt, full


def run(python, wt, script, args, extra=()):
    env = dict(
        os.environ,
        PYTHONPATH=f"{wt}{os.pathsep}{HERE}",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
    )
    env.pop("PYTHONSTARTUP", None)
    cmd = [str(python), str(HERE / script), *args, "--json", *extra]
    r = subprocess.run(
        cmd, cwd=tempfile.gettempdir(), env=env, text=True, capture_output=True
    )
    if r.returncode:
        sys.exit(
            f"{wt} {script} {' '.join(args)}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
        )
    return json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo",
        required=True,
        type=pathlib.Path,
        help="vLLM checkout with an editable build",
    )
    p.add_argument(
        "--python",
        required=True,
        type=pathlib.Path,
        help="interpreter of that build's venv",
    )
    p.add_argument("--base", required=True)
    p.add_argument("--pr", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--scratch",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "vllm-sched-ab",
    )
    p.add_argument("--harness", choices=["decode", "churn", "both"], default="both")
    p.add_argument("--skip-identity", action="store_true")
    a = p.parse_args()
    a.repo = a.repo.resolve()
    a.scratch.mkdir(parents=True, exist_ok=True)
    sides = {}
    for name, sha in (("base", a.base), ("pr", a.pr)):
        sides[name] = worktree(a.repo, a.scratch, sha)
        print(f"{name}: {sides[name][1][:10]} -> {sides[name][0]}")

    if not a.skip_identity:
        print("\n## identity (per-step SchedulerOutput hashes + final KV state)")
        ok = True
        for i, (script, args) in enumerate(IDENTITY, 1):
            hashes = {}
            for name, (wt, _) in sides.items():
                hp = a.scratch / f"h_{name}_{i}.txt"
                out = run(a.python, wt, script, args, ["--hash-out", str(hp)])
                hashes[name] = (out["hash_all"], out["kv_hash"], hp.read_text())
            same = hashes["base"] == hashes["pr"]
            ok &= same
            steps = len(hashes["base"][2].splitlines())
            print(
                f"{'IDENTICAL' if same else 'DIFFER   '} {script} {' '.join(args)} "
                f"({steps} steps, steps {hashes['base'][0]}, kv {hashes['base'][1]})"
            )
        print("identity:", "PASS" if ok else "FAIL")

    plans = [k for k in ("decode", "churn") if a.harness in (k, "both")]
    res = {k: {n: [] for n in sides} for k in plans}
    for r in range(a.rounds):
        order = ["base", "pr"] if r % 2 == 0 else ["pr", "base"]
        print(
            f"\nround {r + 1}/{a.rounds} load {os.getloadavg()[0]:.2f} order {order}",
            flush=True,
        )
        for k in plans:
            for name in order:
                res[k][name].append(
                    [run(a.python, sides[name][0], s, args) for s, args in TIMING[k]]
                )

    def table(k, key_s, key_u, label):
        print(
            f"\n## {k} harness, {a.rounds} alternated rounds, median per side "
            f"(spread = max-min of {key_s})"
        )
        print(f"| config | side | {label} schedule µs | update µs | total | vs base |")
        print("|---|---|---|---|---|---|")
        for j, (_, args) in enumerate(TIMING[k]):
            cfg = " ".join(args[: 4 if k == "decode" else 6])
            tot = {}
            for name in ("base", "pr"):
                ss = [x[j][key_s] for x in res[k][name]]
                uu = [x[j][key_u] for x in res[k][name]]
                tot[name] = (
                    statistics.median(ss),
                    statistics.median(uu),
                    max(ss) - min(ss),
                )
            for name in ("base", "pr"):
                sm, um, sp = tot[name]
                t = sm + um
                tb = tot["base"][0] + tot["base"][1]
                print(
                    f"| {cfg} | {name} | {sm:.0f} (±{sp:.0f}) | {um:.0f} | {t:.0f} "
                    f"| {100 * (t / tb - 1):+.1f}% |"
                )

    for k in plans if a.rounds else ():
        if k == "decode":
            table(k, "sched_med_us", "upd_med_us", "median")
        else:
            table(k, "sched_mean_us", "upd_mean_us", "mean")
    with open(a.scratch / "ab_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nraw: {a.scratch / 'ab_results.json'}")


if __name__ == "__main__":
    main()
