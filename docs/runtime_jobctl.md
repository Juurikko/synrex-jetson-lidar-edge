<!-- SPDX-License-Identifier: CC-BY-4.0 -->
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
environment snapshot, whose hash is recorded and checked before execution.
The exact admitted base mapping is passed explicitly to the worker. The command
receives that base plus the worker-derived reserved lock context described below;
its complete derived environment hash is recorded separately. Native library
changes cannot silently diverge from the admitted base mapping. A changed worker
environment still fails admission.
The wrapper's
own file hash is sealed and checked before and after execution. HEAD is checked
again after execution; GPU jobs also require a clean source checkout afterward.
Use a separate immutable worktree for running jobs; do not develop or write
outputs into that worktree while they run. These endpoint checks detect retained
changes but cannot prove absence of a transient mutation that was later undone.
Commands must
run in the foreground and wait for their children; do not launch daemons or
detach descendants. This is an owned campaign wrapper, not a system scheduler.

Each new worker is a Linux child subreaper. It observes direct-command exit with
`waitid(WNOWAIT)` and retains the original leader until its original process group
has drained. This pins the leader PID and group ID while signals are issued.
After **any** direct-command exit, including SIGKILL, the worker checks member
start ticks, boot and session identity, sends TERM and then KILL when needed,
adopts/reaps ordinary descendants, and finally reaps the leader. A command that
returns zero while leaving live descendants receives a failed receipt. A command that
waits for all its children keeps its normal success behavior. Cleanup allows a
one-second TERM grace and two-second KILL grace; observed wall time is still
charged without clamping.

Already-dead kernel-adopted zombies from the original session may be reaped even
if they changed process group. Their exact PID/start ticks, parent, boot, group,
session and wait status are retained in cleanup proof; no other group is signaled.
When only dead children remain, the direct command's exit code is preserved.
Auxiliary wait status is cleanup evidence, not a scientific or output-quality
verdict. Live alternate-group children and different-session descendants remain
unsupported and fail closed.

Unverifiable process identities, incomplete cleanup, or adopted descendants that
detached into another process group/session produce `CLEANUP_UNVERIFIED` in the
checkpoint. The worker keeps the per-job/GPU descriptors open, writes no terminal
receipt, and waits for operator recovery. Reservation/exclusion remain in force;
SIGTERM does not make an unknown survivor safe. Detached/`setsid` children are
unsupported. A killed worker cannot itself perform cleanup, so inherited
descriptors and an unverified registry entry remain essential.

The durable intent precedes worker spawn. The detached worker holds a per-job
flock, records PID/start ticks/boot identity and child identity, checkpoints, logs
stdout/stderr and publishes an fsynced exit receipt. For GPU jobs it opens the
existing shared lock without `O_CREAT`, verifies the admitted inode, and takes
the established nonblocking exclusive flock. The command inherits both lock
descriptors, preserving resource exclusion if its worker is interrupted. An
unavailable lease fails the attempt rather than competing with another owner.

Python coordinators must explicitly propagate the lock descriptors on every
subprocess edge (Python normally closes them). The worker overlays its derived
`SYNREX_JOB_LOCK_FDS` context only after verifying the original admitted environment.
The JSON context identifies schema `synrex.runtime.lock-fds.v1`, job ID and
per-job/optional GPU descriptor roles with actual descriptor number, device,
inode and path. Caller-supplied reserved context is rejected at launch. Use the
runner's exported API rather than guessing descriptor numbers:

```python
subprocess.run(command, check=True, pass_fds=jobctl.inherited_lock_fds(required=True))
```

`inherited_lock_fds` validates unique roles/descriptors, regular descriptor types
and descriptor/path device/inode identity. Missing or invalid context fails closed;
`required=False` returns an empty tuple only when context is absent. Descendants
must retain the context in their environment and remain in the original process
group. A custom environment must preserve the derived context unchanged. The
context is exclusion plumbing, not permission to launch another job. If both
worker and coordinator die while an ordinary descendant survives, correctly
propagated descriptors keep both locks held; the missing terminal receipt still
blocks a new GPU intent after that descendant exits.

New intents seal the `synrex.runtime.owned-process-group.v2` lifecycle contract in
their immutable specification. Their registry/checkpoint/exit evidence binds the
derived lock context and child environment hash, and accepted terminal receipts
require matching drained-group proof. Previously accepted marker-less historical
receipts retain their original verification contract; adding flags to an old
receipt does not upgrade its scientific or lifecycle evidence.

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
