import asyncio
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from patrickstar_agents.jobs import (
    Jobs,
    native_turn,
    parse_result,
    worker,
)

CAT = {
    "Claude": {"available": True, "default": "claude-opus-5-5", "custom_model": True},
    "Gemini": {
        "available": True,
        "default": "gemini-4-argon-high",
        "models": ["gemini-4-argon-high"],
    },
    "MathOS": {"available": True, "default": "math", "models": ["math"]},
}


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()
        self.jobs = Jobs(Path(self.tmp.name) / "state")
        self.p = patch.object(Jobs, "catalog", return_value=CAT)
        self.p.start()
        self.addCleanup(self.p.stop)

    def start(self, key="request-123", **kw):
        return self.jobs.start(
            provider="Claude",
            prompt="Test",
            workspace=str(self.work),
            request_id=key,
            launch=False,
            **kw,
        )

    def test_idempotent_concurrent_start(self):
        with ThreadPoolExecutor(max_workers=4) as e:
            ids = list(e.map(lambda _: self.start()["id"], range(4)))
        self.assertEqual(len(set(ids)), 1)

    def test_changed_request_refused(self):
        self.start()
        with self.assertRaises(ValueError):
            self.start(model="another")

    def test_unknown_not_resumed(self):
        a = self.start()
        self.jobs.update(a["id"], status="unknown")
        with self.assertRaises(ValueError):
            self.start("continue-123", parent=a["id"])

    def test_native_continuation_serializes_session(self):
        a = self.start()
        self.jobs.update(
            a["id"],
            status="completed",
            native_id="d7de67a1-de7e-4c12-b119-843af28a98f3",
        )
        b = self.start("continue-123", parent=a["id"])
        self.assertEqual(b["native_id"], self.jobs.get(a["id"])["native_id"])
        with self.assertRaises(ValueError):
            self.start("continue-456", parent=a["id"])

    def test_no_resume_without_native_id(self):
        a = self.start()
        self.jobs.update(a["id"], status="completed")
        with self.assertRaises(ValueError):
            self.start("continue-123", parent=a["id"])

    def test_workspace_attachment_escape(self):
        (Path(self.tmp.name) / "secret.txt").write_text("private")
        with self.assertRaises(ValueError):
            self.start(files=["../secret.txt"])
        (self.work / "link").symlink_to(Path(self.tmp.name) / "secret.txt")
        with self.assertRaises(ValueError):
            self.start(files=["link"])

    def test_attachments_snapshotted(self):
        (self.work / "file.txt").write_text("original")
        a = self.start(files=["file.txt"])
        (self.work / "file.txt").write_text("changed")
        self.assertEqual(
            json.loads(self.jobs.get(a["id"])["request"])["files"][0]["text"],
            "original",
        )

    def test_message_idempotence_and_terminal_refusal(self):
        a = self.start()
        self.jobs.message(a["id"], "next", "message-123")
        self.jobs.message(a["id"], "next", "message-123")
        self.assertEqual(self.jobs.status(a["id"])["pending_messages"], 1)
        with self.assertRaises(ValueError):
            self.jobs.message(a["id"], "changed", "message-123")
        self.jobs.update(a["id"], status="completed")
        with self.assertRaises(ValueError):
            self.jobs.message(a["id"], "next", "message-456")

    def test_stop_does_not_signal_arbitrary_pid(self):
        a = self.start()
        self.jobs.update(a["id"], pid=1)
        with patch("os.killpg") as kill:
            self.jobs.stop(a["id"])
            kill.assert_not_called()
        self.assertEqual(self.jobs.get(a["id"])["stop"], 1)

    def test_status_does_not_restart_stale_worker(self):
        a = self.start()
        with self.jobs.connect() as c:
            c.execute(
                "UPDATE jobs SET created=? WHERE id=?", (time.time() - 100, a["id"])
            )
        with patch("subprocess.Popen") as proc:
            self.assertIn("warning", self.jobs.status(a["id"]))
            proc.assert_not_called()

    def test_native_result_and_permission(self):
        self.assertEqual(
            parse_result(
                "Claude", [{"type": "result", "result": "ok", "is_error": False}]
            )[0],
            "completed",
        )
        self.assertEqual(
            parse_result(
                "Claude", [{"type": "result", "permission_denials": [{"tool": "Bash"}]}]
            )[0],
            "awaiting_permission",
        )
        self.assertEqual(
            parse_result(
                "Gemini",
                [
                    {
                        "event": "result",
                        "result": {"status": "SUCCESS", "response": "yes"},
                    }
                ],
            )[0],
            "completed",
        )
        self.assertEqual(parse_result("Gemini", [])[0], "unknown")
        self.assertEqual(
            parse_result(
                "Gemini",
                [{"event": "result", "result": {"status": "SUCCESS", "response": ""}}],
            )[0],
            "unknown",
        )
        self.assertEqual(
            parse_result(
                "Gemini",
                [
                    {
                        "event": "result",
                        "result": {
                            "status": "SUCCESS",
                            "response": "",
                            "denied_actions": [{"action": "write_file"}],
                        },
                    }
                ],
            )[0],
            "awaiting_permission",
        )

    def test_models_not_silently_substituted(self):
        with self.assertRaises(ValueError):
            self.jobs.start(
                provider="Gemini",
                prompt="x",
                workspace=str(self.work),
                request_id="model-123",
                model="other",
                launch=False,
            )

    def test_queued_stop_before_launch(self):
        a = self.start()
        self.jobs.stop(a["id"])

        async def fake(*args):
            return "completed", "should not run", None, ""

        with (
            patch("patrickstar_agents.jobs.load_config"),
            patch("patrickstar_agents.jobs.native_turn", side_effect=fake) as native,
        ):
            asyncio.run(worker(self.jobs, a["id"]))
            # The worker must check queued cancellation before invoking a provider.
            native.assert_not_called()

    def test_page_final_result(self):
        a = self.start()
        self.jobs.update(a["id"], status="completed", result="abcdef")
        self.assertEqual(self.jobs.result(a["id"], 2, 2)["text"], "cd")

    def test_worker_claim_is_exclusive(self):
        a = self.start()
        calls = []

        async def fake(*args):
            calls.append(1)
            await asyncio.sleep(0.03)
            return "completed", "ok", "d7de67a1-de7e-4c12-b119-843af28a98f3", ""

        async def both():
            await asyncio.gather(worker(self.jobs, a["id"]), worker(self.jobs, a["id"]))

        with (
            patch("patrickstar_agents.jobs.load_config"),
            patch("patrickstar_agents.jobs.native_turn", side_effect=fake),
        ):
            asyncio.run(both())
        self.assertEqual(len(calls), 1)

    def test_queued_feedback_resumes_same_session(self):
        a = self.start()
        self.jobs.message(a["id"], "Check result", "feedback-123")
        calls = []

        async def fake(j, jid, req, prompt, cfg, sid):
            calls.append((prompt, sid))
            return "completed", "done", "d7de67a1-de7e-4c12-b119-843af28a98f3", ""

        with (
            patch("patrickstar_agents.jobs.load_config"),
            patch("patrickstar_agents.jobs.native_turn", side_effect=fake),
        ):
            asyncio.run(worker(self.jobs, a["id"]))
        self.assertEqual(len(calls), 2)
        self.assertIn("Check result", calls[1][0])
        self.assertEqual(calls[1][1], "d7de67a1-de7e-4c12-b119-843af28a98f3")
        self.assertEqual(self.jobs.status(a["id"])["pending_messages"], 0)

    def test_real_process_stop_and_deadline(self):
        import sys

        executable = Path(self.tmp.name) / "fake_cli"
        executable.write_text(
            "#!"
            + sys.executable
            + "\nimport time,sys\nsys.stdin.read()\ntime.sleep(30)\n"
        )
        executable.chmod(0o700)
        cfg = SimpleNamespace(claude=SimpleNamespace(executable=str(executable)))
        for stop in (True, False):
            a = self.start("process-" + str(stop))
            req = json.loads(self.jobs.get(a["id"])["request"])
            if not stop:
                with self.jobs.connect() as c:
                    c.execute(
                        "UPDATE jobs SET deadline=? WHERE id=?",
                        (time.time() + 0.05, a["id"]),
                    )

            async def test(a=a, req=req, stop=stop):
                task = asyncio.create_task(
                    native_turn(self.jobs, a["id"], req, "hello", cfg, None)
                )
                if stop:
                    await asyncio.sleep(0.05)
                    self.jobs.stop(a["id"])
                return await asyncio.wait_for(task, 4)

            result = asyncio.run(test())
            self.assertEqual(result[0], "stopped" if stop else "unknown")


if __name__ == "__main__":
    unittest.main()
