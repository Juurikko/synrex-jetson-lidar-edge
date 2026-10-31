# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""CPU fixtures only: temporary git source, registry and simulated GPU lease."""
import importlib.util
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

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def tearDown(self):
        # Never leave fixture children alive on assertion failure.
        if self.controller.registry.exists():
            for job in jobctl.read_json(self.controller.registry)["jobs"]:
                for key in ("child_identity", "worker_identity"):
                    identity = job.get(key)
                    if jobctl.is_live(identity):
                        os.kill(identity["pid"], signal.SIGKILL)
        for worker in self.controller._workers:
            worker.wait(timeout=3)
        self.temporary.cleanup()

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


if __name__ == "__main__":
    unittest.main()
