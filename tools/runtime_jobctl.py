#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Small Linux campaign job runner. See docs/runtime_jobctl.md for its contract."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


STATES = {"STARTING", "RUNNING", "QUEUED", "SUCCEEDED", "FAILED", "CANCELLED",
          "DEFERRED", "NOT_RUN"}
ACTIVE = {"STARTING", "RUNNING", "QUEUED"}
TOTAL_SECONDS = 57600
DEVELOPMENT_SECONDS = 43200


class JobError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    with Path(path).open() as source:
        return json.load(source)


def process_identity(pid=None):
    """PID alone is never evidence: Linux boot and start tick must also match."""
    pid = os.getpid() if pid is None else pid
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return {"pid": int(pid), "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def is_live(identity):
    if not isinstance(identity, dict) or set(identity) != {"pid", "start_ticks", "boot_id"}:
        return False
    try:
        return process_identity(identity["pid"]) == identity
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (OSError, ValueError, IndexError) as error:
        raise JobError("process identity unavailable; cannot reconcile safely") from error


@contextlib.contextmanager
def locked(path):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield handle


def elapsed(job):
    """Same-boot monotonic time; otherwise conservatively retain reservation."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot == job["launch_boot_id"]:
        return max(0, math.ceil(time.monotonic() - job["launch_monotonic"]))
    return max(job["reservation_seconds"], math.ceil(time.time() - job["launch_time"]))


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("absolute job deadline exhausted during preflight")


def file_identity(path, deadline=None):
    check_deadline(deadline)
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise JobError(f"identity input is not a regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            check_deadline(deadline)
            hasher.update(block)
    check_deadline(deadline)
    return {"path": str(path), "sha256": hasher.hexdigest(), "bytes": path.stat().st_size}


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.registry = self.root / "JOB_REGISTRY.json"
        self.registry_lock = self.root / "JOB_REGISTRY.lock"
        self._workers = []

    def reap_owned_workers(self):
        self._workers = [worker for worker in self._workers if worker.poll() is None]

    def _load(self):
        if not self.registry.exists():
            return {"schema_version": 1, "jobs": []}
        value = read_json(self.registry)
        if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
            raise JobError("registry must be an object with a jobs array")
        ids = set()
        for job in value["jobs"]:
            if (job.get("state") not in STATES or job.get("job_id") in ids
                    or job.get("id") != job.get("job_id")):
                raise JobError("invalid or duplicate registry job")
            if job["state"] == "RUNNING" and job.get("identity") != job.get("worker_identity"):
                raise JobError("RUNNING process identity mismatch")
            ids.add(job["job_id"])
        return value

    def _save(self, value):
        value["budget"] = self._budget(value)
        atomic_json(self.registry, value)

    def _job_dir(self, job_id):
        return self.root / "jobs" / job_id

    @staticmethod
    def _budget(value):
        total = development = 0
        for job in value["jobs"]:
            if not job.get("gpu"):
                continue
            if job.get("exit_receipt"):
                charged = job["exit_receipt"]["gpu_seconds"]
            elif job["state"] == "NOT_RUN":
                charged = 0
            else:
                charged = max(job["reservation_seconds"], elapsed(job))
            total += charged
            if job["budget_phase"] == "DEVELOPMENT":
                development += charged
        return {"total_gpu_seconds": total, "development_gpu_seconds": development,
                "total_limit_seconds": TOTAL_SECONDS,
                "development_limit_seconds": DEVELOPMENT_SECONDS,
                "final_reserved_seconds": TOTAL_SECONDS - DEVELOPMENT_SECONDS,
                "budget_qualified": total <= TOTAL_SECONDS and development <= DEVELOPMENT_SECONDS}

    def _verified_receipt(self, job):
        path = self._job_dir(job["job_id"]) / "exit.json"
        if not path.exists():
            return None
        receipt = read_json(path)
        for key in ("job_id", "source_commit", "command_sha256", "budget_phase",
                    "identity_sha256", "worker_identity"):
            if receipt.get(key) != job.get(key):
                raise JobError(f"unverified exit receipt {job['job_id']}: {key}")
        code = receipt.get("exit_code")
        seconds = receipt.get("gpu_seconds")
        if (type(code) is not int or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds < 0
                or (not job["gpu"] and seconds != 0)):
            raise JobError("invalid exit code or GPU charge")
        if not receipt.get("child_reaped") or is_live(job.get("worker_identity")):
            return None
        if is_live(job.get("child_identity")):
            return None
        return receipt

    def _reconcile(self, value):
        for job in value["jobs"]:
            if job["state"] == "NOT_RUN":
                continue
            # Validate all receipts again, including previously terminal jobs.
            receipt = self._verified_receipt(job)
            if receipt:
                job["exit_receipt"] = receipt
                job["state"] = ("CANCELLED" if receipt.get("cancelled") else
                                "SUCCEEDED" if receipt["exit_code"] == 0 else "FAILED")
            elif job.get("exit_receipt"):
                raise JobError(f"durable exit receipt unavailable for {job['job_id']}")
            elif is_live(job.get("worker_identity")) or is_live(job.get("child_identity")):
                # A dead worker does not prove its independently running child stopped.
                job["state"] = "RUNNING"
            elif job["state"] in ACTIVE:
                job["state"] = "DEFERRED"
                job["block_reason"] = (
                    "identity not live and verified exit unavailable; reservation retained"
                )
        return value

    def reconcile(self):
        self.reap_owned_workers()
        with locked(self.registry_lock):
            value = self._reconcile(self._load())
            self._save(value)
            return value

    def launch(self, job_id, command, source_commit, source_dir, configs=(), inputs=(),
               gpu=False, max_seconds=60, phase="DEVELOPMENT", gpu_lock=None,
               gpu_lock_inode=None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", job_id):
            raise JobError("job ID must be a safe, unique path component")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_commit):
            raise JobError("full immutable source commit required")
        if (phase not in {"DEVELOPMENT", "FINAL"} or not math.isfinite(max_seconds)
                or max_seconds <= 0 or not command):
            raise JobError("invalid phase, duration or command")
        if any(not isinstance(arg, str) or not arg for arg in command):
            raise JobError("command must be an argv array of nonempty strings")
        source_dir = str(Path(source_dir).resolve(strict=True))
        actual = subprocess.check_output(["git", "-C", source_dir, "rev-parse", "HEAD"],
                                         text=True).strip()
        if actual != source_commit:
            raise JobError("source HEAD does not match source_commit")
        if gpu:
            dirty = subprocess.check_output(
                ["git", "-C", source_dir, "status", "--porcelain"], text=True
            )
            if dirty:
                raise JobError("GPU jobs require a clean committed source checkout")
            if not gpu_lock or type(gpu_lock_inode) is not int:
                raise JobError("GPU jobs require the established lock path and verified inode")
            lease = Path(gpu_lock).resolve(strict=True)
            if lease.stat().st_ino != gpu_lock_inode or not lease.is_file():
                raise JobError("shared GPU lock inode changed")
            gpu_lock = str(lease)
        identities = {"source_commit": source_commit, "source_dir": source_dir,
                      "command": command, "environment_sha256": digest(dict(os.environ)),
                      "runner_identity": file_identity(__file__),
                      "configs": [file_identity(p) for p in configs],
                      "inputs": [file_identity(p) for p in inputs]}
        if gpu and (not identities["configs"] or not identities["inputs"]):
            raise JobError("GPU jobs require explicit config and input identities")
        with locked(self.registry_lock):
            value = self._reconcile(self._load())
            if any(j["job_id"] == job_id for j in value["jobs"]):
                raise JobError("job ID already exists; use a new ID for a new attempt")
            # Unverified deferred jobs also block new GPU work, even across reboot.
            if gpu and any(j.get("gpu") and j["state"] != "NOT_RUN" and
                           not j.get("exit_receipt") for j in value["jobs"]):
                raise JobError("GPU job has a live intent or unverified exit")
            reservation = math.ceil(max_seconds) + 5 if gpu else 0
            budget = self._budget(value)
            if gpu and (budget["total_gpu_seconds"] + reservation > TOTAL_SECONDS or
                        (phase == "DEVELOPMENT" and
                         budget["development_gpu_seconds"] + reservation > DEVELOPMENT_SECONDS)):
                raise JobError("GPU budget exhausted or final reserve would be consumed")
            job = {"id": job_id, "job_id": job_id, "state": "STARTING", "gpu": bool(gpu),
                   "source_commit": source_commit, "command_sha256": digest(command),
                   "spec_identity": identities, "identity": None,
                   "identity_sha256": digest(identities),
                   "budget_phase": phase, "reservation_seconds": reservation,
                   "max_seconds": max_seconds, "launch_time": time.time(),
                   "launch_monotonic": time.monotonic(),
                   "launch_boot_id": process_identity()["boot_id"],
                   "gpu_lock": gpu_lock, "gpu_lock_inode": gpu_lock_inode,
                   "lease": {"path": gpu_lock, "inode": gpu_lock_inode} if gpu else None,
                   "checkpoint": str(self._job_dir(job_id) / "checkpoint.json"),
                   "config_identity": identities["configs"], "input_identity": identities["inputs"],
                   "allowed_gpu_seconds": min(TOTAL_SECONDS - budget["total_gpu_seconds"],
                       DEVELOPMENT_SECONDS - budget["development_gpu_seconds"]
                       if phase == "DEVELOPMENT" else TOTAL_SECONDS) if gpu else 0}
            directory = self._job_dir(job_id)
            directory.mkdir()
            value["jobs"].append(job)
            self._save(value)  # Durable launch intent MUST precede spawn.
            atomic_json(directory / "intent.json", job)
            try:
                with (directory / "worker.log").open("ab", buffering=0) as log:
                    worker = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--root", str(self.root),
                         "_worker", job_id], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        start_new_session=True, close_fds=True)
            except OSError as error:
                job["state"] = "NOT_RUN"
                job["block_reason"] = str(error)
                self._save(value)
                raise JobError(f"worker spawn failed: {error}") from error
            self._workers.append(worker)
            try:
                job["worker_identity"] = process_identity(worker.pid)
                job["identity"] = job["worker_identity"]
                if not job["worker_identity"]:
                    raise JobError("worker identity unavailable after spawn")
            except (OSError, JobError) as error:
                job["state"] = "DEFERRED"
                job["block_reason"] = "spawn occurred but worker identity unavailable"
                self._save(value)
                raise JobError(job["block_reason"]) from error
            self._save(value)
            return job


def worker_main(controller, job_id):
    directory = controller._job_dir(job_id)
    # The child inherits these descriptors: worker interruption never releases a
    # live child's GPU lease or per-job exclusivity.
    job_fd = os.open(directory / "worker.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(job_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_fd = None
    child = None
    cancelled = False
    error = None
    code = 125
    identity = process_identity()
    with locked(controller.registry_lock):
        value = controller._load()
        job = next(j for j in value["jobs"] if j["job_id"] == job_id)
        if (job.get("worker_identity") != identity
                or job["state"] not in {"STARTING", "RUNNING"}
                or job.get("child_identity")):
            raise JobError("worker identity or launch state mismatch")
        job["state"] = "RUNNING"
        job["identity"] = identity
        controller._save(value)
    atomic_json(directory / "checkpoint.json", {"state": "RUNNING", "worker_identity": identity})
    deadline = job["launch_monotonic"] + job["max_seconds"]

    def git_output(*args):
        check_deadline(deadline)
        return subprocess.check_output(
            ["git", "-C", job["spec_identity"]["source_dir"], *args], text=True,
            timeout=max(.001, deadline - time.monotonic()))

    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        if file_identity(__file__, deadline) != job["spec_identity"]["runner_identity"]:
            raise JobError("job wrapper identity changed before execution")
        # Fail if inputs changed between admission and actual execution.
        for entry in job["spec_identity"]["configs"] + job["spec_identity"]["inputs"]:
            if file_identity(entry["path"], deadline) != entry:
                raise JobError("config/input identity changed before execution")
        actual = git_output("rev-parse", "HEAD").strip()
        if actual != job["source_commit"]:
            raise JobError("source HEAD changed before execution")
        if digest(dict(os.environ)) != job["spec_identity"]["environment_sha256"]:
            raise JobError("execution environment changed before execution")
        if job["gpu"] and git_output("status", "--porcelain"):
            raise JobError("GPU source checkout changed before execution")
        check_deadline(deadline)
        inherited = [job_fd]
        if job["gpu"]:
            # Deliberately no O_CREAT: only the existing controller lease inode.
            gpu_fd = os.open(job["gpu_lock"], os.O_RDONLY | os.O_NOFOLLOW)
            if os.fstat(gpu_fd).st_ino != job["gpu_lock_inode"]:
                raise JobError("GPU lease inode changed before acquisition")
            fcntl.flock(gpu_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.stat(job["gpu_lock"]).st_ino != job["gpu_lock_inode"]:
                raise JobError("GPU lease path changed during acquisition")
            inherited.append(gpu_fd)
        # Recheck after all hashing, Git admission and lease operations. No child
        # may start merely because a reservation once existed before preflight.
        check_deadline(deadline)
        if job["gpu"] and elapsed(job) + 1 >= job["allowed_gpu_seconds"]:
            raise JobError("remaining GPU allowance exhausted before child execution")
        with (directory / "stdout.log").open("ab", buffering=0) as out, \
                (directory / "stderr.log").open("ab", buffering=0) as err:
            child = subprocess.Popen(job["spec_identity"]["command"],
                                     cwd=job["spec_identity"]["source_dir"],
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     start_new_session=True, pass_fds=inherited)
            child_identity = process_identity(child.pid)
            if not child_identity:
                # Fast exit is safe only after wait() reaps this exact child.
                child.poll()
            with locked(controller.registry_lock):
                value = controller._load()
                current = next(j for j in value["jobs"] if j["job_id"] == job_id)
                current["child_identity"] = child_identity
                controller._save(value)
            atomic_json(directory / "checkpoint.json",
                        {"state": "RUNNING", "worker_identity": identity,
                         "child_identity": child_identity})
            while child.poll() is None:
                if cancelled or time.monotonic() >= deadline:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(0.05)
            code = child.wait()
            if not cancelled and time.monotonic() >= deadline:
                code = 124
        for entry in job["spec_identity"]["configs"] + job["spec_identity"]["inputs"]:
            if file_identity(entry["path"], deadline) != entry:
                code = 125
                error = "config/input identity changed during execution"
        if git_output("rev-parse", "HEAD").strip() != job["source_commit"]:
            raise JobError("source HEAD changed during execution")
        if job["gpu"] and git_output("status", "--porcelain"):
            raise JobError("GPU source checkout changed during execution")
        if file_identity(__file__, deadline) != job["spec_identity"]["runner_identity"]:
            raise JobError("job wrapper identity changed during execution")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        code = 124 if isinstance(caught, (TimeoutError, subprocess.TimeoutExpired)) else 125
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    finally:
        if gpu_fd is not None:
            os.close(gpu_fd)
            gpu_fd = None
        charge = elapsed(job) + 1 if job["gpu"] else 0
        budget_qualified = not job["gpu"] or charge <= job["allowed_gpu_seconds"]
        if not budget_qualified:
            error = "actual conservative GPU charge exceeded admitted remaining allowance"
            code = 125
        receipt = {key: job[key] for key in ("job_id", "source_commit", "command_sha256",
                                            "budget_phase", "identity_sha256")}
        receipt.update(worker_identity=identity, exit_code=int(code),
                       gpu_seconds=charge, budget_qualified=budget_qualified,
                       child_reaped=True, cancelled=cancelled, error=error,
                       finished_time=time.time())
        atomic_json(directory / "exit.json", receipt)
        atomic_json(directory / "checkpoint.json", {"state": "EXITED", "exit_receipt": receipt})
        if gpu_fd is not None:
            os.close(gpu_fd)
        os.close(job_fd)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:
        separator = argv.index("--")
        command = argv[separator + 1:]
        argv = argv[:separator]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("reconcile")
    commands.add_parser("status")
    worker = commands.add_parser("_worker")
    worker.add_argument("job_id")
    launch = commands.add_parser("launch")
    launch.add_argument("job_id")
    launch.add_argument("--source-commit", required=True)
    launch.add_argument("--source-dir", required=True)
    launch.add_argument("--config", action="append", default=[])
    launch.add_argument("--input", action="append", default=[])
    launch.add_argument("--gpu", action="store_true")
    launch.add_argument("--gpu-lock")
    launch.add_argument("--gpu-lock-inode", type=int)
    launch.add_argument("--max-seconds", type=float, required=True)
    launch.add_argument("--phase", choices=["DEVELOPMENT", "FINAL"], default="DEVELOPMENT")
    args = parser.parse_args(argv)
    controller = Controller(args.root)
    try:
        if args.action == "_worker":
            return worker_main(controller, args.job_id)
        if args.action == "launch":
            result = controller.launch(args.job_id, command, args.source_commit, args.source_dir,
                                       args.config, args.input, args.gpu, args.max_seconds,
                                       args.phase, args.gpu_lock, args.gpu_lock_inode)
        else:
            result = controller.reconcile()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (JobError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"jobctl: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
