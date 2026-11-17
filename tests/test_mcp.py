"""Exercise the installed MCP server and worker without a model subscription."""

import asyncio
import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path


@asynccontextmanager
async def connect(env):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "patrickstar_agents.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=env,
    )
    request_id = 0

    async def send(message):
        process.stdin.write(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")
        await process.stdin.drain()

    async def request(method, params):
        nonlocal request_id
        request_id += 1
        await send({"id": request_id, "method": method, "params": params})
        async with asyncio.timeout(10):
            while line := await process.stdout.readline():
                response = json.loads(line)
                if "id" not in response:
                    continue  # A server notification.
                if response["id"] != request_id or "error" in response:
                    raise AssertionError(response)
                return response["result"]
        raise AssertionError(f"Server closed stdout during {method}")

    try:
        await request("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "patrickstar-test", "version": "1"},
        })
        await send({"method": "notifications/initialized"})
        yield request
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise


class StdioTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_discovery_and_error_response(self):
        with tempfile.TemporaryDirectory() as directory:
            async with connect({"PATRICKSTAR_EXTERNAL_ROOT": directory}) as request:
                result = await request("tools/list", {})
                tools = {tool["name"]: tool for tool in result["tools"]}
                self.assertTrue(tools["external_status"]["annotations"]["readOnlyHint"])
                self.assertTrue(tools["external_start"]["annotations"]["destructiveHint"])
                reply = await request("tools/call", {
                    "name": "external_list", "arguments": {"limit": 0},
                })
                self.assertTrue(reply["isError"])
                self.assertIn("limit must be", reply["content"][0]["text"])

    async def test_reconnect_and_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "project"
            work.mkdir()
            cli = root / "claude"
            cli.write_text(
                "#!" + sys.executable + "\n"
                "import json, sys, time\n"
                "from pathlib import Path\n"
                "sys.stdin.read()\n"
                "flag = '--resume' if '--resume' in sys.argv else '--session-id'\n"
                "session = sys.argv[sys.argv.index(flag) + 1]\n"
                "with Path('starts').open('a') as f: print(flag, session, file=f)\n"
                "for _ in range(200):\n"
                "    if Path('finish').exists(): break\n"
                "    time.sleep(0.05)\n"
                "print(json.dumps({'type':'result','session_id':session,'result':'done'}))\n"
            )
            cli.chmod(0o700)
            config = root / "config.json"
            config.write_text(json.dumps({"claude": {"executable": str(cli)}}))
            config.chmod(0o600)
            env = {
                "PATRICKSTAR_CONFIG": str(config),
                "PATRICKSTAR_EXTERNAL_ROOT": str(root / "state"),
            }

            async def call(session, name, **args):
                reply = await session("tools/call", {"name": name, "arguments": args})
                self.assertFalse(reply.get("isError"), reply["content"])
                return reply["structuredContent"]

            async def finished(session, job):
                async with asyncio.timeout(15):
                    while True:
                        result = await call(session, "external_result", job_id=job)
                        if result["status"] not in ("queued", "running"):
                            self.assertEqual(result["status"], "completed", result)
                            return result
                        await asyncio.sleep(0.1)

            request = {
                "provider": "Claude",
                "workspace": str(work),
                "prompt": "Run the test",
                "request_id": "test-start-01",
                "timeout_seconds": 30,
            }
            async with connect(env) as first:
                job = (await call(first, "external_start", **request))["id"]
                async with asyncio.timeout(10):
                    while not (work / "starts").exists():
                        await asyncio.sleep(0.1)
                self.assertEqual(
                    (await call(first, "external_status", job_id=job))["status"],
                    "running",
                )

            (work / "finish").touch()
            async with connect(env) as second:
                jobs = (await call(second, "external_list"))["jobs"]
                self.assertEqual([j["id"] for j in jobs], [job])
                result = await finished(second, job)
                self.assertEqual(
                    (await call(second, "external_start", **request))["id"], job
                )
                follow = await call(
                    second,
                    "external_continue",
                    job_id=job,
                    prompt="Continue",
                    request_id="test-continue-01",
                    timeout_seconds=30,
                )
                resumed = await finished(second, follow["id"])
                self.assertEqual(resumed["native_id"], result["native_id"])
            session_id = result["native_id"]
            self.assertEqual(
                (work / "starts").read_text().splitlines(),
                [f"--session-id {session_id}", f"--resume {session_id}"],
            )
