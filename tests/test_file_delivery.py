from __future__ import annotations

import os
import tempfile
import unittest
from asyncio import run
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from ui.file_delivery import (
    DELIVERY_PREFIX,
    FileDeliveryMiddleware,
    clear_delivery_registry,
    lookup_delivery_entry,
    register_file_for_delivery,
    register_files_for_delivery,
)
from ui.file_staging import cleanup_stale_staging


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def _request(
    url: str,
    *,
    method: str = "GET",
    headers=(),
    raw_path: bytes | str | None = None,
):
    sent = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 599,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"passed-through"})

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": method,
        "path": url,
        "raw_path": raw_path if raw_path is not None else url.encode("ascii"),
        "headers": list(headers),
        "query_string": b"",
    }
    await FileDeliveryMiddleware(app)(scope, _receive, send)
    return sent


def _headers(start_message) -> dict[bytes, bytes]:
    return {key.lower(): value for key, value in start_message["headers"]}


class FileDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_delivery_registry()

    def tearDown(self) -> None:
        clear_delivery_registry()

    def test_registers_opaque_relative_url_without_source_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SECRET_WORKSPACE"
            source = root / "SECRET_USER_original.wav"
            root.mkdir()
            source.write_bytes(b"audio-bytes")

            first = register_file_for_delivery(
                source,
                filename="original.wav",
                allowed_roots=(root,),
                ttl_seconds=3600,
            )
            second = register_file_for_delivery(
                source,
                filename="original.wav",
                allowed_roots=(root,),
                ttl_seconds=3600,
            )

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)
            self.assertTrue(str(first).startswith(f"{DELIVERY_PREFIX}/"))
            self.assertNotIn("SECRET_WORKSPACE", str(first))
            self.assertNotIn("SECRET_USER", str(first))
            self.assertNotIn("%", str(first))
            self.assertNotIn("\\", str(first))
            self.assertNotIn(":", str(first))
            self.assertNotIn("file=", str(first))
            self.assertTrue(str(first).endswith("/original.wav"))

    def test_bulk_registration_fails_closed_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.wav"
            second = root / "second.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            def fake_register(path, **kwargs):
                if str(path).endswith("first.wav"):
                    raise RuntimeError("SECRET_WORKSPACE /private/audio/input.wav")
                return "/audiorescue-files/token/second.wav"

            with mock.patch("ui.file_delivery.register_file_for_delivery", side_effect=fake_register):
                urls = register_files_for_delivery(
                    {"original_audio": first, "mixed_audio": second},
                    filenames_by_role={"original_audio": "original.wav", "mixed_audio": "mixed.wav"},
                    allowed_roots=(root,),
                )

            self.assertIsNone(urls["original_audio"])
            self.assertEqual(urls["mixed_audio"], "/audiorescue-files/token/second.wav")

    def test_get_head_and_range_serve_expected_bytes_and_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mixed.wav"
            source.write_bytes(b"0123456789")
            url = register_file_for_delivery(source, filename="mixed.wav", allowed_roots=(root,))
            self.assertIsNotNone(url)

            get_sent = run(_request(str(url)))
            self.assertEqual(get_sent[0]["status"], 200)
            self.assertEqual(get_sent[-1]["body"], b"0123456789")
            headers = _headers(get_sent[0])
            self.assertEqual(headers[b"content-type"], b"audio/wav")
            self.assertEqual(headers[b"content-length"], b"10")
            self.assertEqual(headers[b"accept-ranges"], b"bytes")
            self.assertIn(b'filename="mixed.wav"', headers[b"content-disposition"])

            head_sent = run(_request(str(url), method="HEAD"))
            self.assertEqual(head_sent[0]["status"], 200)
            self.assertEqual(_headers(head_sent[0])[b"content-length"], b"10")
            self.assertEqual(head_sent[-1]["body"], b"")

            range_sent = run(_request(str(url), headers=[(b"range", b"bytes=2-5")]))
            self.assertEqual(range_sent[0]["status"], 206)
            self.assertEqual(range_sent[-1]["body"], b"2345")
            range_headers = _headers(range_sent[0])
            self.assertEqual(range_headers[b"content-range"], b"bytes 2-5/10")
            self.assertEqual(range_headers[b"content-length"], b"4")

            suffix_sent = run(_request(str(url), headers=[(b"range", b"bytes=-3")]))
            self.assertEqual(suffix_sent[0]["status"], 206)
            self.assertEqual(suffix_sent[-1]["body"], b"789")

            string_raw_path_sent = run(_request(str(url), raw_path=str(url)))
            self.assertEqual(string_raw_path_sent[0]["status"], 200)
            self.assertEqual(string_raw_path_sent[-1]["body"], b"0123456789")

    def test_bad_ids_paths_queries_and_ranges_fail_closed_without_path_echo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = "SECRET_WORKSPACE"
            source = root / marker / "source.wav"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            url = register_file_for_delivery(source, filename="original.wav", allowed_roots=(root,))
            self.assertIsNotNone(url)
            token = str(url).split("/")[2]

            bad_urls = [
                f"{DELIVERY_PREFIX}/unknownunknown0000/original.wav",
                f"{DELIVERY_PREFIX}/{token}/wrong.wav",
                f"{DELIVERY_PREFIX}/{token}/../original.wav",
                f"{DELIVERY_PREFIX}/{token}/%2e%2e/original.wav",
                f"{DELIVERY_PREFIX}/{token}/%252e%252e/original.wav",
                f"{DELIVERY_PREFIX}/{token}/C:\\secret\\original.wav",
            ]
            for bad_url in bad_urls:
                with self.subTest(bad_url=bad_url):
                    sent = run(_request(bad_url, raw_path=bad_url.encode("ascii")))
                    self.assertIn(sent[0]["status"], {404, 410})
                    body = sent[-1]["body"].decode("utf-8", errors="ignore")
                    self.assertNotIn(marker, body)
                    self.assertNotIn(str(root), body)

            query_scope_url = str(url)
            sent = []

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http",
                "method": "GET",
                "path": query_scope_url,
                "raw_path": query_scope_url.encode("ascii"),
                "headers": [],
                "query_string": b"path=/secret",
            }
            run(FileDeliveryMiddleware(lambda scope, receive, send: None)(scope, _receive, send))
            self.assertEqual(sent[0]["status"], 404)

            invalid_range = run(_request(str(url), headers=[(b"range", b"bytes=50-60")]))
            self.assertEqual(invalid_range[0]["status"], 416)
            self.assertNotIn(str(root).encode("utf-8"), invalid_range[-1]["body"])

    def test_expired_deleted_directory_and_symlink_replacement_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original.wav"
            source.write_bytes(b"audio")
            url = register_file_for_delivery(
                source,
                filename="original.wav",
                allowed_roots=(root,),
                ttl_seconds=1,
                now=100.0,
            )
            self.assertIsNotNone(url)
            with mock.patch("ui.file_delivery.time.time", return_value=200.0):
                expired = run(_request(str(url)))
            self.assertEqual(expired[0]["status"], 410)

            deleted = root / "deleted.wav"
            deleted.write_bytes(b"deleted")
            deleted_url = register_file_for_delivery(deleted, filename="mixed.wav", allowed_roots=(root,))
            deleted.unlink()
            self.assertEqual(run(_request(str(deleted_url)))[0]["status"], 410)

            directory = root / "directory.wav"
            directory.write_bytes(b"file")
            directory_url = register_file_for_delivery(
                directory, filename="full.wav", allowed_roots=(root,)
            )
            directory.unlink()
            directory.mkdir()
            self.assertEqual(run(_request(str(directory_url)))[0]["status"], 410)

            if hasattr(os, "symlink"):
                target = root / "target.wav"
                target.write_bytes(b"target")
                link = root / "link.wav"
                link.write_bytes(b"link")
                link_url = register_file_for_delivery(
                    link, filename="waveform.png", allowed_roots=(root,)
                )
                link.unlink()
                os.symlink(target, link)
                self.assertEqual(run(_request(str(link_url)))[0]["status"], 410)

    def test_staging_cleanup_invalidates_delivery_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            staging_root = Path(tmp) / "ui-stage"
            session = staging_root / ("a" * 32)
            session.mkdir(parents=True)
            source = session / "original.wav"
            source.write_bytes(b"audio")
            old_time = 100.0
            os.utime(session, (old_time, old_time))
            url = register_file_for_delivery(
                source,
                filename="original.wav",
                allowed_roots=(staging_root,),
                ttl_seconds=3600,
                now=old_time,
            )
            self.assertIsNotNone(lookup_delivery_entry(str(url)))

            cleanup_stale_staging(
                staging_root=staging_root,
                ttl_seconds=60,
                now=old_time + 3600,
            )

            self.assertIsNone(lookup_delivery_entry(str(url)))
            self.assertEqual(run(_request(str(url)))[0]["status"], 404)

    def test_repeated_and_concurrent_reads_do_not_cross_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "job-a" / "mixed.wav"
            second = root / "job-b" / "mixed.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first-job")
            second.write_bytes(b"second-job")
            first_url = register_file_for_delivery(first, filename="mixed.wav", allowed_roots=(root,))
            second_url = register_file_for_delivery(second, filename="mixed.wav", allowed_roots=(root,))
            self.assertNotEqual(first_url, second_url)

            def fetch(url: str) -> bytes:
                return run(_request(url))[-1]["body"]

            urls = [str(first_url), str(second_url)] * 10
            with ThreadPoolExecutor(max_workers=4) as pool:
                bodies = list(pool.map(fetch, urls))

            self.assertEqual(bodies[0::2], [b"first-job"] * 10)
            self.assertEqual(bodies[1::2], [b"second-job"] * 10)


if __name__ == "__main__":
    unittest.main()
