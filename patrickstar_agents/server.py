"""Codex-facing MCP entry point; stdout is reserved for the MCP protocol."""

import asyncio
import json
import os
import time
from importlib.metadata import version

from zeromcp import McpServer

from .jobs import ACTIVE, ROOT, Jobs

os.umask(0o077)
jobs = Jobs(
    os.environ.get("PATRICKSTAR_EXTERNAL_ROOT", str(ROOT)),
    os.environ.get("PATRICKSTAR_CONFIG"),
)
mcp = McpServer(
    "Patrickstar external agents",
    version=version("patrickstar-agents"),
    instructions="""
Delegate bounded work to external native Claude Code or Antigravity agents, or
consult MathOS. Codex remains the coordinator. Only use when the user has asked
for delegation/multiple models. Choose exact models via external_catalog.
Pass the user-authorized workspace, task, acceptance criteria, resource limits,
and remaining deadline. These are native agents, not one-step JSON generators.
Start is asynchronous and idempotent with request_id. Save the returned job ID;
query/wait that ID, never resubmit because a wait timed out. Continue completed
jobs to preserve their provider conversation. Unknown outcomes require inspection
of files/logs/remote jobs, never automatic replay. Messages are delivered at the
next native turn boundary, not immediate interruption. Native permission checks
remain enabled; access intent is not a filesystem sandbox guarantee.
Parallel tasks may share a directory: assign separate files or separate checkouts
where they can conflict. Do not infer verification from a model's claim; inspect
actual changes and tests. MathOS is a text consultant and cannot itself run tools.
""",
)
mcp.tools.redact_exceptions = True
read_tool = mcp.tool(read_only=True, destructive=False, idempotent=True, open_world=False)
write_tool = mcp.tool(read_only=False, destructive=True, idempotent=True, open_world=True)


@read_tool
def external_catalog() -> dict:
    """List configured providers, exact model IDs and capabilities. No model calls."""
    return jobs.catalog()


@write_tool
def external_start(
    provider: str,
    prompt: str,
    workspace: str,
    request_id: str,
    model: str = "",
    effort: str = "",
    access: str = "work",
    timeout_seconds: int = 3600,
    files: list[str] | None = None,
) -> dict:
    """Start native Claude/Gemini work or MathOS consultation and immediately return a durable job ID.

    request_id must be unique per intended task; reuse it only to recover the same
    submission. workspace is an absolute user-authorized directory. access is
    work or read_only (native plan mode). files are optional relative UTF-8 files,
    limited to 100KB combined; MathOS receives only prompt and these attachments.
    timeout_seconds is the execution deadline (30–43200), not the polling timeout.
    Native tools and network follow the provider's permissions; no bypass flags.
    """
    return jobs.start(
        provider=provider,
        prompt=prompt,
        workspace=workspace,
        request_id=request_id,
        model=model,
        effort=effort,
        access=access,
        timeout_seconds=timeout_seconds,
        files=files,
    )


@read_tool
async def external_status(
    job_id: str, after_event: int = 0, wait_seconds: int = 0
) -> dict:
    """Read progress, or wait at most 20s for a new event/completion. Waiting never stops the agent.

    Pass next_event from the previous response as after_event to avoid repeated
    output. If still running, wait/check later; do not submit a replacement.
    """
    if not 0 <= wait_seconds <= 20 or after_event < 0:
        raise ValueError("Invalid wait/cursor")
    end = time.monotonic() + wait_seconds
    while True:
        value = jobs.status(job_id, after_event)
        if value["events"] or value["status"] not in ACTIVE or time.monotonic() >= end:
            return value
        await asyncio.sleep(min(0.5, max(0, end - time.monotonic())))


@read_tool
def external_result(job_id: str, offset: int = 0, limit: int = 12000) -> dict:
    """Read the saved final/last-turn answer with pagination; does not invoke a model."""
    return jobs.result(job_id, offset, limit)


@write_tool
def external_continue(
    job_id: str,
    prompt: str,
    request_id: str,
    timeout_seconds: int = 3600,
    files: list[str] | None = None,
) -> dict:
    """Resume the latest completed job in its exact native conversation and workspace.

    Returns a new durable job ID. Refuses active, failed, stopped or unknown jobs
    to avoid replaying uncertain effects. MathOS receives explicit stored history.
    No automatic change of model, provider, access or billing route.
    """
    req = json.loads(jobs.get(job_id)["request"])
    return jobs.start(
        provider=req["provider"],
        model=req["model"],
        effort=req["effort"],
        access=req["access"],
        workspace=req["workspace"],
        prompt=prompt,
        request_id=request_id,
        timeout_seconds=timeout_seconds,
        files=files,
        parent=job_id,
    )


@write_tool
def external_message(job_id: str, text: str, request_id: str) -> dict:
    """Queue coordinator feedback for the next native turn without interrupting current work.

    Uses the original job deadline. Delivery may wait until the current native
    turn ends. For an already completed task use external_continue instead.
    """
    return jobs.message(job_id, text, request_id)


@write_tool
def external_stop(job_id: str) -> dict:
    """Request stopping this job's own local agent process. Does not undo edits or cancel remote Slurm jobs."""
    return jobs.stop(job_id)


@read_tool
def external_list(limit: int = 10) -> dict:
    """List recent external jobs for recovery after Codex reconnects. No provider calls."""
    if not 1 <= limit <= 50:
        raise ValueError("limit must be 1–50")
    with jobs.connect() as c:
        rows = c.execute(
            "SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
    return {
        "jobs": [
            {k: v for k, v in jobs.status(r["id"]).items() if k != "events"}
            for r in rows
        ]
    }


def main():
    asyncio.run(mcp.stdio_async())


if __name__ == "__main__":
    main()
