#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Small Linux campaign job runner. See docs/runtime_jobctl.md for its contract."""

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
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
LIFECYCLE_CONTRACT = "synrex.runtime.owned-process-group.v2"
LOCK_FDS_ENV = "SYNREX_JOB_LOCK_FDS"
LOCK_FDS_SCHEMA = "synrex.runtime.lock-fds.v1"


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


def inherited_lock_fds(required=True):
    """Validate worker-derived inherited descriptors for explicit pass_fds.

    This is resource-exclusion plumbing, not authorization for another job.
    A coordinator must preserve this environment context on its descendants.
    """
    raw = os.environ.get(LOCK_FDS_ENV)
    if raw is None and not required:
        return ()
    try:
        context = json.loads(raw)
        if (set(context) != {"schema", "job_id", "entries"}
                or context["schema"] != LOCK_FDS_SCHEMA
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", context["job_id"])
                or not isinstance(context["entries"], list)
                or not 1 <= len(context["entries"]) <= 2):
            raise ValueError("invalid lock context")
        roles, descriptors = set(), []
        for entry in context["entries"]:
            if (set(entry) != {"role", "fd", "device", "inode", "path"}
                    or entry["role"] not in {"job", "gpu"} or entry["role"] in roles
                    or type(entry["fd"]) is not int or not 3 <= entry["fd"] < 2**31
                    or entry["fd"] in descriptors
                    or type(entry["device"]) is not int or type(entry["inode"]) is not int
                    or not isinstance(entry["path"], str)
                    or not Path(entry["path"]).is_absolute()):
                raise ValueError("invalid lock descriptor entry")
            info, path_info = os.fstat(entry["fd"]), os.stat(entry["path"], follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or not stat.S_ISREG(path_info.st_mode)
                    or (info.st_dev, info.st_ino) != (entry["device"], entry["inode"])
                    or (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino)):
                raise ValueError("inherited lock identity changed")
            roles.add(entry["role"])
            descriptors.append(entry["fd"])
        if "job" not in roles:
            raise ValueError("per-job lock required")
        return tuple(descriptors)
    except (TypeError, ValueError, OSError, KeyError) as error:
        raise JobError("valid inherited job lock context required") from error


def enable_subreaper():
    # Linux per-process setting; no host policy or global scheduler changes.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise JobError("cannot enable owned worker child subreaper")


def process_details(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    try:
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "state": fields[0], "parent_pid": int(fields[1]),
                "process_group": int(fields[2]), "session": int(fields[3]),
                "start_ticks": int(fields[19])}
    except (ValueError, IndexError) as error:
        raise JobError("process group identity unavailable") from error


def group_members(group):
    """Inspect the original group while its unreaped leader pins the PGID."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    leader = process_details(group["pgid"])
    if (boot != group["boot_id"] or leader is None
            or leader["start_ticks"] != group["leader_start_ticks"]
            or leader["process_group"] != group["pgid"]
            or leader["session"] != group["pgid"]):
        raise JobError("original process group leader identity unavailable")
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        member = process_details(int(path.name))
        if member is None or member["process_group"] != group["pgid"]:
            continue
        if (member["session"] != group["pgid"]
                or member["start_ticks"] < group["leader_start_ticks"]):
            raise JobError("process group member identity mismatch")
        members.append(member)
    # Adopted children in another session/group are unsupported, not proof of an
    # empty job. Never signal an unrelated or unidentified process group.
    for pid in Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split():
        member = process_details(int(pid))
        if member is not None and member["process_group"] != group["pgid"]:
            raise JobError("detached descendant unsupported; cleanup unverified")
    return members


def drain_process_group(child, group):
    """Terminate/reap ordinary descendants before reaping the pinned leader.

    The direct child must never be poll()/wait() reaped before this function.
    Every signal targets an original, identity-checked group with its original
    leader still present, preventing a recycled PGID from becoming a target.
    """
    initial = group_members(group)
    proof = {"group": group, "drained": False,
             "initial_descendants": [m for m in initial if m["pid"] != child.pid],
             "reaped_descendants": [], "signals": []}
    for signum, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        members = group_members(group)
        if any(m["state"] != "Z" for m in members):
            os.killpg(group["pgid"], signum)
            proof["signals"].append(int(signum))
        limit = time.monotonic() + grace
        while True:
            members = group_members(group)
            for member in members:
                if member["pid"] != child.pid and member["parent_pid"] == os.getpid():
                    pid, status = os.waitpid(member["pid"], os.WNOHANG)
                    if pid:
                        proof["reaped_descendants"].append(dict(member, wait_status=status))
            members = group_members(group)
            if len(members) == 1 and members[0]["pid"] == child.pid and members[0]["state"] == "Z":
                code = child.wait()
                # Nothing live could fork between that snapshot and leader reap.
                # If a reused group appears now, retain exclusion without signals.
                try:
                    os.killpg(group["pgid"], 0)
                except ProcessLookupError:
                    proof["drained"] = True
                    return code, proof
                raise JobError("process group still exists after original leader reap")
            if time.monotonic() >= limit:
                break
            time.sleep(.01)
    raise JobError("original process group did not drain; exclusion retained")


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
        if digest(job["spec_identity"]) != job["identity_sha256"]:
            raise JobError("unverified immutable job specification digest")
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
        if (job.get("lifecycle_contract") or job["spec_identity"].get("lifecycle_contract")
                or receipt.get("lifecycle_contract")):
            proof = receipt.get("process_group_cleanup")
            if (job.get("lifecycle_contract") != LIFECYCLE_CONTRACT
                    or job["spec_identity"].get("lifecycle_contract") != LIFECYCLE_CONTRACT
                    or receipt.get("lifecycle_contract") != LIFECYCLE_CONTRACT
                    or not isinstance(proof, dict) or proof.get("drained") is not True
                    or proof.get("group") != job.get("child_group")
                    or receipt.get("lock_fd_context") != job.get("lock_fd_context")
                    or receipt.get("child_environment_sha256")
                    != job.get("child_environment_sha256")):
                raise JobError("unverified original process group cleanup or lock context")
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
        # os.putenv/native extensions can change the C environment without updating
        # os.environ. Hash and explicitly inherit the same Python snapshot.
        execution_environment = dict(os.environ)
        if LOCK_FDS_ENV in execution_environment:
            raise JobError("reserved inherited lock context cannot be supplied by caller")
        identities = {"source_commit": source_commit, "source_dir": source_dir,
                      "command": command, "environment_sha256": digest(execution_environment),
                      "lifecycle_contract": LIFECYCLE_CONTRACT,
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
                   "lifecycle_contract": LIFECYCLE_CONTRACT,
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
                        start_new_session=True, close_fds=True, env=execution_environment)
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
    child_group = None
    cleanup = {"group": None, "drained": True, "initial_descendants": [],
               "reaped_descendants": [], "signals": []}
    lock_context = child_environment_sha256 = None
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
        enable_subreaper()
        if file_identity(__file__, deadline) != job["spec_identity"]["runner_identity"]:
            raise JobError("job wrapper identity changed before execution")
        # Fail if inputs changed between admission and actual execution.
        for entry in job["spec_identity"]["configs"] + job["spec_identity"]["inputs"]:
            if file_identity(entry["path"], deadline) != entry:
                raise JobError("config/input identity changed before execution")
        actual = git_output("rev-parse", "HEAD").strip()
        if actual != job["source_commit"]:
            raise JobError("source HEAD changed before execution")
        execution_environment = dict(os.environ)
        if digest(execution_environment) != job["spec_identity"]["environment_sha256"]:
            raise JobError("execution environment changed before execution")
        if LOCK_FDS_ENV in execution_environment:
            raise JobError("reserved inherited lock context in worker base environment")
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
        entries = []
        locks = [("job", job_fd, directory / "worker.lock")]
        if gpu_fd is not None:
            locks.append(("gpu", gpu_fd, job["gpu_lock"]))
        for role, fd, path in locks:
            info = os.fstat(fd)
            entries.append({"role": role, "fd": fd, "device": info.st_dev,
                            "inode": info.st_ino, "path": str(path)})
        lock_context = {"schema": LOCK_FDS_SCHEMA, "job_id": job_id, "entries": entries}
        child_environment = dict(execution_environment,
                                 **{LOCK_FDS_ENV: canonical(lock_context).decode()})
        child_environment_sha256 = digest(child_environment)
        with locked(controller.registry_lock):
            value = controller._load()
            current = next(j for j in value["jobs"] if j["job_id"] == job_id)
            current.update(lock_fd_context=lock_context,
                           child_environment_sha256=child_environment_sha256)
            controller._save(value)
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
                                     start_new_session=True, pass_fds=inherited,
                                     env=child_environment)
            cleanup = None  # Spawn occurred: no terminal receipt without group proof.
            details = process_details(child.pid)
            if (details is None or details["process_group"] != child.pid
                    or details["session"] != child.pid):
                raise JobError("new command process group identity unavailable")
            child_group = {"pgid": child.pid, "leader_start_ticks": details["start_ticks"],
                           "boot_id": identity["boot_id"]}
            child_identity = {"pid": child.pid, "start_ticks": details["start_ticks"],
                              "boot_id": identity["boot_id"]}
            with locked(controller.registry_lock):
                value = controller._load()
                current = next(j for j in value["jobs"] if j["job_id"] == job_id)
                current["child_identity"] = child_identity
                current["child_group"] = child_group
                controller._save(value)
            atomic_json(directory / "checkpoint.json",
                        {"state": "RUNNING", "worker_identity": identity,
                         "child_identity": child_identity, "child_group": child_group,
                         "lock_fd_context": lock_context,
                         "child_environment_sha256": child_environment_sha256})
            command_exited = False
            while True:
                # Retain the exact leader as a zombie after exit until its group
                # is drained; poll()/wait() here would allow unsafe PGID reuse.
                if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                    command_exited = True
                    break
                if cancelled or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            code, cleanup = drain_process_group(child, child_group)
            if command_exited and cleanup["initial_descendants"]:
                error = "command exited with surviving descendants; original group drained"
                if code == 0:
                    code = 125
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
    finally:
        if child is not None and not (cleanup and cleanup["drained"]):
            try:
                if child_group is None:
                    raise JobError("spawned command group identity unavailable")
                _, cleanup = drain_process_group(child, child_group)
            except Exception as caught:
                # No accepted terminal evidence and no voluntary lease release.
                # Unknown/detached survivors require operator reconciliation.
                blocked = {"state": "CLEANUP_UNVERIFIED", "worker_identity": identity,
                           "child_group": child_group, "error": str(caught),
                           "lock_fd_context": lock_context}
                try:
                    atomic_json(directory / "checkpoint.json", blocked)
                    with locked(controller.registry_lock):
                        value = controller._load()
                        current = next(j for j in value["jobs"] if j["job_id"] == job_id)
                        current["cleanup_unverified"] = blocked
                        controller._save(value)
                except Exception as publication_error:
                    print(f"cleanup checkpoint unavailable: {publication_error}", file=sys.stderr)
                while True:
                    time.sleep(1)
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
                       lifecycle_contract=LIFECYCLE_CONTRACT, process_group_cleanup=cleanup,
                       lock_fd_context=lock_context,
                       child_environment_sha256=child_environment_sha256,
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
