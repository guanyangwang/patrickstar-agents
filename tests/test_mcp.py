"""Exercise the installed MCP server and worker without a model subscription."""

import asyncio
import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def connect(params):
    async with (
        stdio_client(params) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        await session.initialize()
        yield session


class ReconnectTest(unittest.IsolatedAsyncioTestCase):
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
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "patrickstar_agents.server"],
                env={
                    "PATRICKSTAR_CONFIG": str(config),
                    "PATRICKSTAR_EXTERNAL_ROOT": str(root / "state"),
                },
            )

            async def call(session, name, **args):
                reply = await session.call_tool(name, args)
                self.assertFalse(reply.isError, reply.content)
                return reply.structuredContent or json.loads(reply.content[0].text)

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
            async with connect(params) as first:
                job = (await call(first, "external_start", **request))["id"]
                async with asyncio.timeout(10):
                    while not (work / "starts").exists():
                        await asyncio.sleep(0.1)
                self.assertEqual(
                    (await call(first, "external_status", job_id=job))["status"],
                    "running",
                )

            (work / "finish").touch()
            async with connect(params) as second:
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
