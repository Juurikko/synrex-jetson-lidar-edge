<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Campaign job receipts

`tools/runtime_jobctl.py` is a Linux, standard-library-only owned job wrapper.
It writes an object with a `jobs` array to `ROOT/JOB_REGISTRY.json`. Use a new
job ID for every attempt. `id` and `job_id` are identical unique IDs. `identity`
is the verified worker PID/start-tick/boot identity (null before spawn), while
`spec_identity` holds immutable source/command/environment/config/input evidence.
`lease` identifies the existing shared lock path/inode; `checkpoint` points to
the durable per-job checkpoint. Reconciliation precedes every launch and also runs
through `status` or `reconcile`. Never edit this registry while workers run.

```sh
python tools/runtime_jobctl.py --root /admitted/campaign launch unique-job \
  --source-dir /committed/source --source-commit FULL_COMMIT \
  --config /admitted/config.json --input /admitted/data.manifest \
  --gpu --gpu-lock /established/host-gpu.lock --gpu-lock-inode VERIFIED_INODE \
  --phase DEVELOPMENT --max-seconds 600 -- /absolute/program argument
python tools/runtime_jobctl.py --root /admitted/campaign reconcile
```

CPU jobs omit `--gpu` and its lease arguments. GPU admission requires clean,
committed source and explicit config/input hashes. Identify the actual inputs,
or a previously verified immutable dataset manifest; a manifest does not itself
prove its referenced files are unchanged. Commands inherit the launching
environment, whose hash is recorded and checked before execution. The wrapper's
own file hash is sealed and checked before and after execution. HEAD is checked
again after execution; GPU jobs also require a clean source checkout afterward.
Use a separate immutable worktree for running jobs; do not develop or write
outputs into that worktree while they run. These endpoint checks detect retained
changes but cannot prove absence of a transient mutation that was later undone.
Commands must
run in the foreground and wait for their children; do not launch daemons or
detach descendants. This is an owned campaign wrapper, not a system scheduler.

The durable intent precedes worker spawn. The detached worker holds a per-job
flock, records PID/start ticks/boot identity and child identity, checkpoints, logs
stdout/stderr and publishes an fsynced exit receipt. For GPU jobs it opens the
existing shared lock without `O_CREAT`, verifies the admitted inode, and takes
the established nonblocking exclusive flock. The command inherits both lock
descriptors, preserving resource exclusion if its worker is interrupted. An
unavailable lease fails the attempt rather than competing with another owner.

An absolute deadline applies to worker preflight as well as execution. Chunked
identity hashing and bounded Git subprocesses check that deadline; a final check
after lease acquisition prevents an expired worker from spawning the command.
Operating-system I/O or scheduling stalls can overrun a deadline before control
returns. Actual charges are never clamped: an overrun of admitted remaining
allowance forces a failed receipt with `budget_qualified: false`, and aggregate
budget qualification is also recorded explicitly.

Wall time from launch intent, including startup, is conservatively charged for
GPU attempts. The requested deadline plus five seconds is reserved before
spawn; a successful receipt replaces that reservation with rounded-up measured
wall time plus one second of publication margin. All development charges/reservations must fit 43,200 seconds and all
phases together must fit 57,600 seconds, preserving 14,400 seconds for FINAL.
Unverified interrupted jobs retain at least their reservation and continue
accruing same-boot elapsed time. Across boots they retain the larger reservation
or wall-clock interval. This intentionally overcharges interruption rather than
claiming unobserved GPU idle time. CPU receipts charge zero GPU seconds.

Reconciliation never treats a live matching worker/child as finished. A stale,
reused or unavailable identity without a matching durable exit is `DEFERRED`,
retains budget, and blocks further GPU admission. Automatic retries are forbidden.
Missing or mismatched terminal receipts fail closed. SIGTERM/SIGINT requests
cancel the child process group; deadline expiry is exit 124. Worker loss can
leave a child running, so an operator must recover evidence before any new GPU
job; restarting the CLI does not magically reconstruct an unobserved exit.

Run the CPU-only temporary-registry fixtures with:

```sh
python -m unittest discover -s tests -p test_runtime_jobctl.py -v
```
