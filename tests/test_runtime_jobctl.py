# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""CPU fixtures only: temporary git source, registry and simulated GPU lease."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "runtime_jobctl", Path(__file__).resolve().parents[1] / "tools/runtime_jobctl.py")
jobctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jobctl)


class JobControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Own orphaned fixture processes after deliberate worker loss, so this
        # test process can reap them instead of relying on the host init process.
        jobctl.enable_subreaper()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime-jobctl-fixture-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.git("init", "-q")
        (self.source / "fixture.txt").write_text("CPU test source\n")
        self.git("add", ".")
        self.git("-c", "user.name=CPU fixture", "-c", "user.email=fixture@invalid",
                 "commit", "-qm", "test-only fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.controller = jobctl.Controller(self.root / "test-only-registry")
        self.lease = self.root / "test-only-gpu.lock"
        self.lease.touch()
        self.config = self.root / "config.json"
        self.config.write_text('{"fixture": "CPU only"}')
        self.input = self.root / "input.txt"
        self.input.write_text("CPU fixture input")
        self.descendants = []

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def tearDown(self):
        # Never leave fixture children alive on assertion failure.
        for identity in self.descendants:
            if jobctl.is_live(identity):
                os.kill(identity["pid"], signal.SIGKILL)
        if self.controller.registry.exists():
            for job in jobctl.read_json(self.controller.registry)["jobs"]:
                for key in ("child_identity", "worker_identity"):
                    identity = job.get(key)
                    if jobctl.is_live(identity):
                        os.kill(identity["pid"], signal.SIGKILL)
        for worker in self.controller._workers:
            worker.wait(timeout=3)
        limit = time.monotonic() + 3
        while time.monotonic() < limit:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not pid:
                time.sleep(.01)
        self.temporary.cleanup()

    def wait_marker(self, path, timeout=3):
        limit = time.monotonic() + timeout
        while not path.exists():
            self.assertLess(time.monotonic(), limit, "fixture marker deadline")
            time.sleep(.01)
        value = json.loads(path.read_text())
        self.descendants.append(value["identity"])
        return value

    def descendant_script(self, marker, *, parent_wait=True, ignore_term=False,
                          propagate=False, detached=False, new_process_group=False):
        runner = str(Path(jobctl.__file__).resolve())
        loader = (
            "import importlib.util; "
            f"s=importlib.util.spec_from_file_location('fixture_jobctl',{runner!r}); "
            "j=importlib.util.module_from_spec(s); s.loader.exec_module(j); "
        )
        child = (
            "import json,os,signal,time; from pathlib import Path; " + loader +
            ("os.setpgid(0,0); " if new_process_group else "") +
            ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else "") +
            f"Path({str(marker)!r}).write_text(json.dumps({{'identity':j.process_identity(),"
            "'process_group':os.getpgrp(),"
            f"'lock_fds':list(j.inherited_lock_fds()) if {propagate!r} else []}})); "
            "time.sleep(30)"
        )
        return (
            "import subprocess,sys,time; from pathlib import Path; " + loader +
            f"c=subprocess.Popen([sys.executable,'-c',{child!r}],"
            f"pass_fds=j.inherited_lock_fds() if {propagate!r} else (),"
            f"start_new_session={detached!r}); "
            f"marker=Path({str(marker)!r}); "
            "exec('while not marker.exists(): time.sleep(.005)'); " +
            ("c.wait()" if parent_wait else "sys.exit(0)")
        )

    def assert_fake_locks_held(self, job_id):
        for path in [self.lease, self.controller._job_dir(job_id) / "worker.lock"]:
            with path.open() as handle:
                with self.assertRaises(BlockingIOError):
                    jobctl.fcntl.flock(handle, jobctl.fcntl.LOCK_EX | jobctl.fcntl.LOCK_NB)

    def launch(self, job_id, script="print('fixture done')", **kwargs):
        return self.controller.launch(job_id, [sys.executable, "-c", script],
                                      self.commit, self.source, **kwargs)

    def simulated_gpu(self, job_id, **kwargs):
        return self.launch(job_id, gpu=True, configs=[self.config], inputs=[self.input],
                           gpu_lock=str(self.lease), gpu_lock_inode=self.lease.stat().st_ino,
                           **kwargs)

    def wait(self, job_id, states=None, timeout=5):
        if states is None:
            states = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.controller.reconcile()
            job = next(j for j in value["jobs"] if j["job_id"] == job_id)
            if job["state"] in states:
                return job
            time.sleep(.02)
        self.fail(f"fixture did not reach {states}: {job}")

    def wait_for_child(self, job_id, timeout=3):
        deadline = time.monotonic() + timeout
        while True:
            value = self.controller.reconcile()
            job = next(j for j in value["jobs"] if j["job_id"] == job_id)
            if job.get("child_identity"):
                return job
            self.assertIn(job["state"], jobctl.ACTIVE, job)
            self.assertLess(time.monotonic(), deadline, job)
            time.sleep(.02)

    def test_verified_exit_and_duplicate_rejection(self):
        self.launch("once")
        job = self.wait("once")
        self.assertEqual(job["state"], "SUCCEEDED", job)
        self.assertEqual(job["id"], job["job_id"])
        self.assertEqual(job["identity"], job["worker_identity"])
        self.assertIn("command", job["spec_identity"])
        self.assertEqual(
            job["checkpoint"], str(self.controller._job_dir("once") / "checkpoint.json")
        )
        receipt = job["exit_receipt"]
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["gpu_seconds"], 0)
        self.assertEqual(receipt["source_commit"], self.commit)
        self.assertTrue(receipt["budget_qualified"])
        self.assertEqual(len(receipt["command_sha256"]), 64)
        self.assertTrue((self.controller._job_dir("once") / "intent.json").exists())
        with self.assertRaisesRegex(jobctl.JobError, "already exists"):
            self.launch("once")

    def test_nonzero_and_timeout_are_real_failures(self):
        self.launch("nonzero", "import sys; sys.exit(7)")
        job = self.wait("nonzero")
        self.assertEqual(job["exit_receipt"]["exit_code"], 7, job)
        self.launch("timeout", "import time; time.sleep(20)", max_seconds=.15)
        job = self.wait("timeout")
        self.assertEqual(job["exit_receipt"]["exit_code"], 124, job)

    def test_native_environment_divergence_uses_admitted_snapshot(self):
        variable = "SYNREX_JOBCTL_ENVIRONMENT_FIXTURE"
        with mock.patch.dict(os.environ, {variable: "admitted"}):
            # Native libraries and os.putenv can leave os.environ stale. The
            # child must receive the exact mapping whose hash was admitted.
            os.putenv(variable, "unrecorded-native-value")
            self.launch("environment-snapshot", f"import os; print(os.environ[{variable!r}])")
            job = self.wait("environment-snapshot")
        self.assertEqual(job["state"], "SUCCEEDED", job)
        self.assertEqual(
            (self.controller._job_dir("environment-snapshot") / "stdout.log").read_text(),
            "admitted\n",
        )

    def test_worker_environment_tampering_fails_before_child(self):
        marker = self.root / "must-not-execute"
        original_popen = subprocess.Popen

        def tampered_worker(command, *args, **kwargs):
            if "_worker" in command:
                environment = dict(kwargs.get("env", os.environ))
                environment["SYNREX_JOBCTL_TAMPER_FIXTURE"] = "not-admitted"
                kwargs["env"] = environment
            return original_popen(command, *args, **kwargs)

        with mock.patch.object(jobctl.subprocess, "Popen", side_effect=tampered_worker):
            self.launch("environment-tampered",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()")
        job = self.wait("environment-tampered")
        self.assertEqual(job["state"], "FAILED", job)
        self.assertEqual(job["exit_receipt"]["exit_code"], 125, job)
        self.assertIn("execution environment changed", job["exit_receipt"]["error"], job)
        self.assertFalse(marker.exists(), job)

    def test_caller_exit_leaves_durable_worker(self):
        result = subprocess.run([
            sys.executable, str(Path(jobctl.__file__)), "--root", str(self.controller.root),
            "launch", "detached", "--source-commit", self.commit,
            "--source-dir", str(self.source), "--max-seconds", "5", "--",
            sys.executable, "-c", "import time; time.sleep(.2); print('durable')"],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.wait("detached")["state"], "SUCCEEDED")

    def test_preflight_deadline_prevents_child_launch(self):
        marker = self.root / "must-not-execute"
        original_popen = subprocess.Popen

        def delayed_worker(command, *args, **kwargs):
            if "_worker" in command:
                time.sleep(.15)
            return original_popen(command, *args, **kwargs)

        with mock.patch.object(jobctl.subprocess, "Popen", side_effect=delayed_worker):
            self.launch("expired-before-child",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()", max_seconds=.05)
        job = self.wait("expired-before-child")
        self.assertEqual(job["exit_receipt"]["exit_code"], 124)
        self.assertIn("deadline", job["exit_receipt"]["error"])
        self.assertFalse(marker.exists())

    def test_interrupted_worker_keeps_live_child_and_gpu_exclusivity(self):
        self.simulated_gpu("interrupted", script="import time; time.sleep(.7)", max_seconds=5)
        job = self.wait_for_child("interrupted")
        self.assertEqual(job["identity"], job["worker_identity"])
        self.assertEqual(job["lease"], {"path": str(self.lease), "inode": self.lease.stat().st_ino})
        self.assertEqual(job["config_identity"], job["spec_identity"]["configs"])
        os.kill(job["worker_identity"]["pid"], signal.SIGKILL)
        time.sleep(.05)
        self.assertEqual(self.controller.reconcile()["jobs"][0]["state"], "RUNNING")
        with self.assertRaisesRegex(jobctl.JobError, "live intent or unverified exit"):
            self.simulated_gpu("duplicate-resource")
        with self.lease.open() as handle:
            with self.assertRaises(BlockingIOError):
                jobctl.fcntl.flock(handle, jobctl.fcntl.LOCK_EX | jobctl.fcntl.LOCK_NB)
        deferred = self.wait("interrupted", {"DEFERRED"})
        self.assertNotIn("exit_receipt", deferred)
        self.assertGreaterEqual(self.controller.reconcile()["budget"]["total_gpu_seconds"], 10)
        with self.assertRaises(jobctl.JobError):
            self.simulated_gpu("still-blocked")

    def test_controlled_interrupt_restart_reconcile_and_new_attempt(self):
        self.simulated_gpu("cancel-request", script="import time; time.sleep(10)", max_seconds=15)
        job = self.wait_for_child("cancel-request")
        os.kill(job["worker_identity"]["pid"], signal.SIGTERM)
        cancelled = self.wait("cancel-request", {"CANCELLED"})
        self.assertTrue(cancelled["exit_receipt"]["cancelled"])
        self.assertNotEqual(cancelled["exit_receipt"]["exit_code"], 0)
        self.assertFalse(jobctl.is_live(cancelled["child_identity"]))
        self.controller.reap_owned_workers()
        # Reconstruct solely from durable registry/receipts as a new CLI would.
        self.controller = jobctl.Controller(self.controller.root)
        recovered = self.controller.reconcile()["jobs"][0]
        self.assertEqual(recovered["state"], "CANCELLED")
        self.simulated_gpu("fresh-attempt-after-cancel")
        self.assertEqual(self.wait("fresh-attempt-after-cancel")["state"], "SUCCEEDED")

    def test_source_mutation_during_child_is_failed(self):
        self.simulated_gpu("source-mutates", script="import time; time.sleep(.3)", max_seconds=5)
        self.wait_for_child("source-mutates")
        (self.source / "fixture.txt").write_text("changed after command began\n")
        job = self.wait("source-mutates")
        self.assertEqual(job["state"], "FAILED")
        self.assertEqual(job["exit_receipt"]["exit_code"], 125)
        self.assertIn("source checkout changed during execution", job["exit_receipt"]["error"])

    def test_stale_pid_reuse_and_missing_identity_fail_closed(self):
        self.assertFalse(jobctl.is_live({"pid": os.getpid(), "start_ticks": -1,
                                        "boot_id": jobctl.process_identity()["boot_id"]}))
        self.launch("stale", "import time; time.sleep(.4)")
        # A lost controller identity must never manufacture a successful receipt.
        with jobctl.locked(self.controller.registry_lock):
            value = self.controller._load()
            value["jobs"][0]["worker_identity"]["start_ticks"] = -1
            value["jobs"][0].pop("child_identity", None)
            self.controller._save(value)
        self.assertEqual(self.controller.reconcile()["jobs"][0]["state"], "DEFERRED")

    def test_budget_reserve_and_total_exhaustion_before_spawn(self):
        # Admission checks use CPU commands and a temporary fixture lease only.
        with mock.patch.object(jobctl, "DEVELOPMENT_SECONDS", 6), \
                mock.patch.object(jobctl, "TOTAL_SECONDS", 14):
            with self.assertRaisesRegex(jobctl.JobError, "reserve"):
                self.simulated_gpu("dev-rejected", max_seconds=2)
            self.assertFalse(self.controller.registry.exists())
            self.simulated_gpu("final", phase="FINAL", max_seconds=2)
            self.wait("final")
            with self.assertRaisesRegex(jobctl.JobError, "budget"):
                self.simulated_gpu("total-rejected", phase="FINAL", max_seconds=20)

    def test_starting_intent_blocks_other_gpu_and_tampered_receipt_fails(self):
        self.simulated_gpu("receipt")
        job = self.wait("receipt")
        path = self.controller._job_dir("receipt") / "exit.json"
        receipt = jobctl.read_json(path)
        receipt["source_commit"] = "0" * 40
        jobctl.atomic_json(path, receipt)
        with self.assertRaisesRegex(jobctl.JobError, "unverified exit"):
            self.controller.reconcile()
        receipt["source_commit"] = self.commit
        jobctl.atomic_json(path, receipt)
        # Inject a launch-intent-only fixture to reproduce crash-before-spawn.
        with jobctl.locked(self.controller.registry_lock):
            value = self.controller._load()
            pending = dict(job)
            pending.update(id="intent-only", job_id="intent-only", state="STARTING", identity=None)
            for key in ("worker_identity", "child_identity", "exit_receipt"):
                pending.pop(key, None)
            value["jobs"].append(pending)
            self.controller._save(value)
        self.assertEqual(self.controller.reconcile()["jobs"][1]["state"], "DEFERRED")
        with self.assertRaisesRegex(jobctl.JobError, "unverified exit"):
            self.simulated_gpu("intent-blocked")

    def test_missing_shared_lock_is_not_created(self):
        self.lease.unlink()
        with self.assertRaises(FileNotFoundError):
            self.controller.launch("no-lock", [sys.executable, "-c", "pass"],
                                   self.commit, self.source, [self.config], [self.input],
                                   True, 1, "DEVELOPMENT", str(self.lease), 123)
        self.assertFalse(self.lease.exists())

    def test_normal_command_waiting_for_child_has_drained_proof(self):
        self.simulated_gpu("normal-wait", script=(
            "import subprocess,sys; "
            "c=subprocess.Popen([sys.executable,'-c','print(42)']); c.wait()"), max_seconds=5)
        job = self.wait("normal-wait")
        self.assertEqual(job["state"], "SUCCEEDED", job)
        self.assertTrue(job["exit_receipt"]["process_group_cleanup"]["drained"])
        self.assertEqual(job["exit_receipt"]["process_group_cleanup"]["initial_descendants"], [])

    def test_direct_command_sigkill_drains_waited_child_and_unrelated_group_survives(self):
        marker = self.root / "grandchild.json"
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     start_new_session=True)
        try:
            self.simulated_gpu("command-killed", script=self.descendant_script(marker),
                               max_seconds=10)
            descendant = self.wait_marker(marker)
            job = self.wait_for_child("command-killed")
            self.assert_fake_locks_held("command-killed")
            os.kill(job["child_identity"]["pid"], signal.SIGKILL)
            terminal = self.wait("command-killed")
            self.assertEqual(terminal["state"], "FAILED", terminal)
            self.assertEqual(terminal["exit_receipt"]["exit_code"], -signal.SIGKILL)
            proof = terminal["exit_receipt"]["process_group_cleanup"]
            self.assertTrue(proof["drained"])
            self.assertIn(descendant["identity"]["pid"],
                          [entry["pid"] for entry in proof["reaped_descendants"]])
            self.assertFalse(jobctl.is_live(descendant["identity"]))
            self.assertIsNone(unrelated.poll(), "unrelated process group was signalled")
            with self.lease.open() as handle:
                jobctl.fcntl.flock(handle, jobctl.fcntl.LOCK_EX | jobctl.fcntl.LOCK_NB)
        finally:
            unrelated.kill()
            unrelated.wait()

    def test_exit_zero_with_survivor_is_failed_and_survivor_reaped(self):
        marker = self.root / "survivor.json"
        self.simulated_gpu("unwaited", script=self.descendant_script(marker, parent_wait=False),
                           max_seconds=10)
        descendant = self.wait_marker(marker)
        terminal = self.wait("unwaited")
        self.assertEqual(terminal["state"], "FAILED", terminal)
        self.assertEqual(terminal["exit_receipt"]["exit_code"], 125)
        self.assertIn("surviving descendants", terminal["exit_receipt"]["error"])
        self.assertFalse(jobctl.is_live(descendant["identity"]))

    def test_term_ignoring_descendant_is_killed_on_worker_interrupt(self):
        marker = self.root / "ignores-term.json"
        self.simulated_gpu("ignores-term", script=self.descendant_script(marker, ignore_term=True),
                           max_seconds=10)
        descendant = self.wait_marker(marker)
        job = self.wait_for_child("ignores-term")
        os.kill(job["worker_identity"]["pid"], signal.SIGTERM)
        terminal = self.wait("ignores-term")
        self.assertEqual(terminal["state"], "CANCELLED", terminal)
        self.assertIn(int(signal.SIGKILL),
                      terminal["exit_receipt"]["process_group_cleanup"]["signals"])
        self.assertFalse(jobctl.is_live(descendant["identity"]))

    def test_worker_and_command_loss_preserve_inherited_descendant_exclusion(self):
        marker = self.root / "inherited-locks.json"
        self.simulated_gpu("dual-loss", script=self.descendant_script(marker, propagate=True),
                           max_seconds=10)
        descendant = self.wait_marker(marker)
        job = self.wait_for_child("dual-loss")
        self.assertEqual(len(descendant["lock_fds"]), 2)
        self.assertEqual(descendant["process_group"], job["child_group"]["pgid"])
        os.kill(job["worker_identity"]["pid"], signal.SIGKILL)
        os.kill(job["child_identity"]["pid"], signal.SIGKILL)
        self.wait("dual-loss", {"DEFERRED"})
        self.assertTrue(jobctl.is_live(descendant["identity"]))
        self.assert_fake_locks_held("dual-loss")
        with self.assertRaisesRegex(jobctl.JobError, "unverified exit"):
            self.simulated_gpu("dual-loss-blocked")
        self.assertFalse((self.controller._job_dir("dual-loss") / "exit.json").exists())
        os.kill(descendant["identity"]["pid"], signal.SIGKILL)
        os.waitpid(descendant["identity"]["pid"], 0)
        with self.lease.open() as handle:
            jobctl.fcntl.flock(handle, jobctl.fcntl.LOCK_EX | jobctl.fcntl.LOCK_NB)
        self.assertNotIn("exit_receipt", self.controller.reconcile()["jobs"][0])

    def test_unverifiable_group_retains_worker_and_locks_without_receipt(self):
        copied = self.source / "test_only_runner.py"
        raw = Path(jobctl.__file__).read_text().replace(
            'if __name__ == "__main__":',
            'def group_members(group):\n    raise JobError("fixture malformed process scan")\n\n'
            'if __name__ == "__main__":')
        copied.write_text(raw)
        self.git("add", ".")
        self.git("-c", "user.name=CPU fixture", "-c", "user.email=fixture@invalid",
                 "commit", "-qm", "test-only malformed scan fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        spec = importlib.util.spec_from_file_location("test_only_bad_scan", copied)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "dont_write_bytecode", True):
            spec.loader.exec_module(module)
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.controller = module.Controller(self.controller.root)
        self.simulated_gpu("bad-scan", max_seconds=5)
        checkpoint = self.controller._job_dir("bad-scan") / "checkpoint.json"
        limit = time.monotonic() + 3
        while (not checkpoint.exists()
               or jobctl.read_json(checkpoint)["state"] != "CLEANUP_UNVERIFIED"):
            self.assertLess(time.monotonic(), limit)
            time.sleep(.01)
        self.assertEqual(self.controller.reconcile()["jobs"][0]["state"], "RUNNING")
        self.assertGreaterEqual(self.controller.reconcile()["budget"]["total_gpu_seconds"], 10)
        self.assert_fake_locks_held("bad-scan")
        self.assertFalse((self.controller._job_dir("bad-scan") / "exit.json").exists())
        with self.assertRaisesRegex(jobctl.JobError, "unverified exit"):
            # Different module defines a distinct exception class.
            try:
                self.simulated_gpu("bad-scan-blocked")
            except module.JobError as error:
                raise jobctl.JobError(str(error)) from error

    def test_new_receipt_requires_bound_group_and_lock_context(self):
        self.simulated_gpu("bound-proof")
        job = self.wait("bound-proof")
        path = self.controller._job_dir("bound-proof") / "exit.json"
        receipt = jobctl.read_json(path)
        self.assertEqual(receipt["lock_fd_context"], job["lock_fd_context"])
        self.assertEqual(receipt["child_environment_sha256"], job["child_environment_sha256"])
        receipt["process_group_cleanup"]["drained"] = False
        jobctl.atomic_json(path, receipt)
        with self.assertRaisesRegex(jobctl.JobError, "process group cleanup"):
            self.controller.reconcile()

    def test_existing_legacy_terminal_receipt_remains_valid(self):
        self.launch("legacy-fixture")
        job = self.wait("legacy-fixture")
        # Reconstruct the former marker-less intent/receipt representation.
        job.pop("lifecycle_contract")
        job["spec_identity"].pop("lifecycle_contract")
        job["identity_sha256"] = jobctl.digest(job["spec_identity"])
        receipt = dict(job["exit_receipt"], identity_sha256=job["identity_sha256"])
        for key in ["lifecycle_contract", "process_group_cleanup", "lock_fd_context",
                    "child_environment_sha256"]:
            receipt.pop(key)
        path = self.controller._job_dir("legacy-fixture") / "exit.json"
        jobctl.atomic_json(path, receipt)
        self.assertEqual(self.controller._verified_receipt(job), receipt)

    def test_stripping_new_intent_markers_cannot_downgrade_receipt(self):
        self.launch("no-downgrade")
        job = self.wait("no-downgrade")
        path = self.controller._job_dir("no-downgrade") / "exit.json"
        receipt = dict(job["exit_receipt"])
        job.pop("lifecycle_contract")
        job["spec_identity"].pop("lifecycle_contract")
        receipt.pop("lifecycle_contract")
        receipt.pop("process_group_cleanup")
        jobctl.atomic_json(path, receipt)
        with self.assertRaisesRegex(jobctl.JobError, "specification digest"):
            self.controller._verified_receipt(job)

    def test_detached_descendant_is_explicitly_unverified_and_excluded(self):
        marker = self.root / "detached-descendant.json"
        self.simulated_gpu("unsupported-detach", script=self.descendant_script(
            marker, parent_wait=False, detached=True, propagate=True), max_seconds=10)
        descendant = self.wait_marker(marker)
        checkpoint = self.controller._job_dir("unsupported-detach") / "checkpoint.json"
        limit = time.monotonic() + 3
        while (not checkpoint.exists()
               or jobctl.read_json(checkpoint)["state"] != "CLEANUP_UNVERIFIED"):
            self.assertLess(time.monotonic(), limit)
            time.sleep(.01)
        self.assertIn("detached descendant unsupported", jobctl.read_json(checkpoint)["error"])
        self.assertTrue(jobctl.is_live(descendant["identity"]))
        self.assert_fake_locks_held("unsupported-detach")
        self.assertFalse((self.controller._job_dir("unsupported-detach") / "exit.json").exists())

    def test_reserved_environment_and_lock_context_validation(self):
        with mock.patch.dict(os.environ, {jobctl.LOCK_FDS_ENV: "not worker derived"}):
            with self.assertRaisesRegex(jobctl.JobError, "reserved"):
                self.launch("caller-spoofed")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(jobctl.inherited_lock_fds(required=False), ())
            with self.assertRaises(jobctl.JobError):
                jobctl.inherited_lock_fds()
        with self.lease.open() as handle:
            info = os.fstat(handle.fileno())
            entry = {"role": "job", "fd": handle.fileno(), "device": info.st_dev,
                     "inode": info.st_ino, "path": str(self.lease)}
            context = {"schema": jobctl.LOCK_FDS_SCHEMA, "job_id": "fixture", "entries": [entry]}
            def validate(value):
                with mock.patch.dict(os.environ, {jobctl.LOCK_FDS_ENV: json.dumps(value)}):
                    return jobctl.inherited_lock_fds()
            self.assertEqual(validate(context), (handle.fileno(),))
            cases = [dict(context, schema="unknown"), dict(context, job_id="../bad"),
                     dict(context, entries=[entry, entry]),
                     dict(context, entries=[entry, dict(entry, role="gpu")]),
                     dict(context, entries=[dict(entry, inode=-1)]),
                     dict(context, entries=[dict(entry, fd=True)]),
                     dict(context, entries=[dict(entry, fd=10**100)]),
                     dict(context, entries=[dict(entry, path=str(self.root))])]
            for value in cases:
                with self.subTest(value=value), self.assertRaises(jobctl.JobError):
                    validate(value)
        with self.assertRaises(jobctl.JobError):
            validate(context)  # Closed descriptor identity is never accepted.
        descriptor = os.open(self.root, os.O_DIRECTORY | os.O_RDONLY)
        try:
            info = os.fstat(descriptor)
            context["entries"] = [dict(entry, fd=descriptor, device=info.st_dev,
                                       inode=info.st_ino, path=str(self.root))]
            with self.assertRaises(jobctl.JobError):
                validate(context)
        finally:
            os.close(descriptor)

    def test_malformed_process_identity_cannot_become_group_proof(self):
        with mock.patch.object(jobctl.Path, "read_text", return_value="malformed"):
            with self.assertRaises(jobctl.JobError):
                jobctl.process_details(os.getpid())

    def zombie_script(self, marker, code, detached=False):
        # Foreground coordinator waits for its direct child. That child forks an
        # auxiliary process, which exits after changing only its process group.
        # The worker adopts the already-dead auxiliary, matching the gh edge.
        auxiliary = "\n".join([
            "import json,os,time",
            "from pathlib import Path",
            "if os.fork() == 0:",
            "    " + ("os.setsid()" if detached else "os.setpgid(0,0)"),
            "    raw=Path('/proc/self/stat').read_text()",
            "    fields=raw[raw.rfind(')')+2:].split()",
            "    identity={'pid':os.getpid(),'start_ticks':int(fields[19]),",
            "              'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}",
            f"    Path({str(marker)!r}).write_text(json.dumps({{'identity':identity,",
            "          'process_group':os.getpgrp(),'session':os.getsid(0)}))",
            "    os._exit(23)",
            "os._exit(0)",
        ])
        return "\n".join([
            "import json,subprocess,sys,time",
            "from pathlib import Path",
            f"child=subprocess.Popen([sys.executable,'-c',{auxiliary!r}])",
            "assert child.wait() == 0",
            f"marker=Path({str(marker)!r})",
            "while not marker.exists(): time.sleep(.005)",
            "identity=json.loads(marker.read_text())['identity']",
            "while True:",
            "    raw=Path('/proc/'+str(identity['pid'])+'/stat').read_text()",
            "    if raw[raw.rfind(')')+2:].split()[0] == 'Z': break",
            "    time.sleep(.005)",
            f"sys.exit({code})",
        ])

    def test_dead_adopted_same_session_alternate_group_is_reaped_with_original_code(self):
        for code in (0, 7):
            with self.subTest(code=code):
                name = f"owned-zombie-{code}"
                marker = self.root / (name + ".json")
                self.simulated_gpu(name, script=self.zombie_script(marker, code), max_seconds=10)
                descendant = self.wait_marker(marker)
                job = self.wait(name)
                receipt = job["exit_receipt"]
                self.assertEqual(receipt["exit_code"], code, receipt)
                self.assertEqual(job["state"], "SUCCEEDED" if code == 0 else "FAILED", job)
                proof = receipt["process_group_cleanup"]
                self.assertTrue(proof["drained"])
                self.assertEqual(proof["signals"], [], "no live process needed a signal")
                records = [p for p in proof["reaped_descendants"]
                           if p["pid"] == descendant["identity"]["pid"]]
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(record["state"], "Z")
                self.assertEqual(record["parent_pid"], job["worker_identity"]["pid"])
                self.assertEqual(record["start_ticks"], descendant["identity"]["start_ticks"])
                self.assertEqual(record["boot_id"], descendant["identity"]["boot_id"])
                self.assertEqual(record["process_group"], descendant["process_group"])
                self.assertNotEqual(record["process_group"], proof["group"]["pgid"])
                self.assertEqual(record["session"], proof["group"]["pgid"])
                self.assertEqual(record["wait_status"], 23 << 8)
                self.assertIsNone(jobctl.process_details(descendant["identity"]["pid"]))

    def test_live_same_session_alternate_group_retains_exclusion(self):
        marker = self.root / "live-alternate-group.json"
        self.simulated_gpu("live-alternate", script=self.descendant_script(
            marker, parent_wait=False, propagate=True, new_process_group=True), max_seconds=10)
        descendant = self.wait_marker(marker)
        checkpoint = self.controller._job_dir("live-alternate") / "checkpoint.json"
        limit = time.monotonic() + 3
        while (not checkpoint.exists()
               or jobctl.read_json(checkpoint)["state"] != "CLEANUP_UNVERIFIED"):
            self.assertLess(time.monotonic(), limit)
            time.sleep(.01)
        self.assertTrue(jobctl.is_live(descendant["identity"]))
        self.assert_fake_locks_held("live-alternate")
        self.assertFalse((self.controller._job_dir("live-alternate") / "exit.json").exists())

    def test_dead_different_session_descendant_still_retains_exclusion(self):
        marker = self.root / "dead-detached-session.json"
        self.simulated_gpu("dead-detached", script=self.zombie_script(marker, 0, detached=True),
                           max_seconds=10)
        descendant = self.wait_marker(marker)
        checkpoint = self.controller._job_dir("dead-detached") / "checkpoint.json"
        limit = time.monotonic() + 3
        while (not checkpoint.exists()
               or jobctl.read_json(checkpoint)["state"] != "CLEANUP_UNVERIFIED"):
            self.assertLess(time.monotonic(), limit)
            time.sleep(.01)
        details = jobctl.process_details(descendant["identity"]["pid"])
        self.assertEqual(details["state"], "Z")
        self.assertNotEqual(details["session"],
                            jobctl.read_json(checkpoint)["child_group"]["pgid"])
        self.assert_fake_locks_held("dead-detached")
        self.assertFalse((self.controller._job_dir("dead-detached") / "exit.json").exists())

    def test_changed_adopted_zombie_identity_is_not_reaped(self):
        group = {"pgid": 10, "leader_start_ticks": 5, "boot_id": "fixture-boot"}
        leader = {"pid": 10, "state": "Z", "parent_pid": os.getpid(),
                  "start_ticks": 5, "process_group": 10, "session": 10}
        zombie = dict(leader, pid=11, process_group=11, start_ticks=6)
        changed = dict(zombie, start_ticks=7)
        with mock.patch.object(jobctl, "group_members", return_value=[leader, zombie]), \
                mock.patch.object(jobctl, "process_details", return_value=changed), \
                mock.patch.object(jobctl.os, "waitpid") as wait, \
                mock.patch.object(jobctl.os, "killpg") as kill:
            with self.assertRaisesRegex(jobctl.JobError, "identity changed before reap"):
                jobctl.drain_process_group(mock.Mock(pid=10), group)
        wait.assert_not_called()
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
