# SPDX-License-Identifier: MIT
"""Bounded HTTP response contract for the FRACTAL ZIP range reader."""

import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "fetch_fractal", Path(__file__).resolve().parents[1] / "tools" / "fetch_fractal.py"
)
assert spec is not None and spec.loader is not None
fetch_fractal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_fractal)


class Response(io.BytesIO):
    status = 206

    def __init__(self, data, content_range):
        super().__init__(data)
        self.headers = {"Content-Range": content_range}
        self.read_lengths = []

    def read(self, size=-1):
        self.read_lengths.append(size)
        return super().read(size)


class TestFetchFractal(unittest.TestCase):
    def test_accepts_exact_range_and_reads_at_most_one_extra_byte(self):
        response = Response(b"abc", "bytes 10-12/100")
        with patch.object(fetch_fractal.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertEqual(fetch_fractal.fetch("https://example.test/archive", 10, 3), b"abc")
        self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=10-12")
        self.assertEqual(response.read_lengths, [4])

    def test_rejects_shifted_or_invalid_range_before_reading_body(self):
        for content_range in ("bytes 11-13/100", "bytes 10-12/12", ""):
            with self.subTest(content_range=content_range):
                responses = [Response(b"abc", content_range) for _ in range(5)]
                with (
                    patch.object(fetch_fractal.urllib.request, "urlopen", side_effect=responses),
                    patch.object(fetch_fractal.time, "sleep"),
                    self.assertRaisesRegex(ValueError, "unexpected Content-Range"),
                ):
                    fetch_fractal.fetch("https://example.test/archive", 10, 3)
                self.assertTrue(all(r.read_lengths == [] for r in responses))

    def test_rejects_extra_body_bytes_with_bounded_read(self):
        responses = [Response(b"abcdef", "bytes 10-12/100") for _ in range(5)]
        with (
            patch.object(fetch_fractal.urllib.request, "urlopen", side_effect=responses),
            patch.object(fetch_fractal.time, "sleep"),
            self.assertRaisesRegex(ValueError, "incorrect HTTP range length"),
        ):
            fetch_fractal.fetch("https://example.test/archive", 10, 3)
        self.assertTrue(all(r.read_lengths == [4] for r in responses))
