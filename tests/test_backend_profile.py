# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Actual CPU TCP fixture must never fabricate GPU phase measurements."""

import json
import subprocess

import pytest
from test_generic_live import FIXTURE, SESSION
from test_manifest_live import pipeline


@pytest.mark.parametrize("timing", ["enabled", "disabled"])
@pytest.mark.parametrize("wire", ["binary-v3", "binary-source-v1"])
def test_backend_profile_opaque_cpu_is_explicitly_unsupported(tmp_path, timing, wire):
    results, _ = pipeline(tmp_path, timing=timing, wire=wire, profile_backend="enabled")
    assert all(value["code"] == 0 for value in results.values()), results
    receiver = results["receive"]["receipt"]
    assert receiver["backend_profiling"] == "enabled"
    trial = receiver["trial_backend_profile"]
    assert trial["availability"] == "UNSUPPORTED_BACKEND"
    assert trial["observed_returned_windows"] == 5
    assert trial["observed_returned_frame_points"] == 34
    for profile in [trial, *(row["backend_profile"] for row in receiver["sources"])]:
        assert profile["backend"] is None and profile["workload"] is None
        assert all(value is None for value in profile["phase_seconds"].values())
        assert all(value is None for value in profile["counters"].values())
        assert profile["storage_bytes"] <= 4096
        assert len(json.dumps(profile).encode()) <= profile["maximum_receipt_bytes"]
    assert [row["backend_profile"]["observed_returned_windows"]
            for row in receiver["sources"]] == [3, 2]
    assert receiver["timing_instrumentation"] == timing
    if timing == "disabled":
        assert receiver["trial_timings"] == {}
        assert all(row["timings"] is None for row in receiver["sources"])
    for role in ("send", "subscribe"):
        assert "trial_backend_profile" not in results[role]["receipt"]


@pytest.mark.parametrize("setting", [None, "disabled"])
def test_backend_profile_off_default_keeps_existing_receipt_and_completion(tmp_path, setting):
    results, _ = pipeline(tmp_path, profile_backend=setting)
    assert all(value["code"] == 0 for value in results.values()), results
    assert "trial_backend_profile" not in results["receive"]["receipt"]
    assert all("backend_profile" not in row
               for row in results["receive"]["receipt"]["sources"])


@pytest.mark.parametrize("failure", ["inference", "late-decode", "output-map"])
def test_backend_profile_retains_failed_prefix_without_inventing_samples(tmp_path, failure):
    results, _ = pipeline(tmp_path, failure=failure, profile_backend="enabled",
                          sources=1 if failure == "output-map" else 2)
    receiver = results["receive"]["receipt"]
    assert results["receive"]["code"] == 1 and receiver["status"] == "FAIL"
    trial = receiver["trial_backend_profile"]
    assert trial["availability"] == "UNSUPPORTED_BACKEND"
    assert trial["observed_returned_windows"] == sum(
        row["backend_profile"]["observed_returned_windows"] for row in receiver["sources"])
    assert trial["observed_returned_frame_points"] == receiver["freshly_classified_records"]
    if failure == "inference":
        assert trial["observed_returned_windows"] == 0
    if failure == "late-decode":
        assert trial["observed_returned_windows"] == 3
    if failure == "output-map":
        # Subscriber map fails after the receiver has emitted the full finite source.
        assert trial["observed_returned_windows"] == 3
        assert receiver["emitted_records"] == 23
        assert receiver["subscriber_snapshot_acknowledged_records"] == 0


def test_backend_profile_invalid_option_fails_before_package_or_backend(tmp_path):
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [str(FIXTURE), "receive", "--wire", "binary-v3", "--model", str(tmp_path / "missing"),
         "--receipt", str(receipt), "--session", SESSION,
         "--manifest", str(tmp_path / "missing-manifest"), "--profile-backend", "yes"],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 1
    assert "profile-backend requires enabled or disabled" in json.loads(result.stdout)["error"]
    assert json.loads(result.stderr.splitlines()[-1])["initializations"] == 0
