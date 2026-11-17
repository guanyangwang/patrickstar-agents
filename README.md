# PatrickStar Agents

Let Codex delegate work to Claude Code and Gemini through MCP. Each agent runs in the project folder you choose. You can close the MCP client, reconnect later, and continue the same agent conversation.

## Install

On Intel and Apple Silicon Macs, install Python 3.11+, Codex CLI and either Claude Code (`claude`) or Antigravity CLI (`agy`). Sign in to the agent you want to use and clone the repository.

```sh
git clone https://github.com/guanyangwang/patrickstar-agents.git
cd patrickstar-agents
python3 -m venv .venv
.venv/bin/python -m pip install .
codex mcp add patrickstar_agents -- "$PWD/.venv/bin/patrickstar-agents"
```

If a previous install failed while building `cryptography`, run `git pull --ff-only` followed by `.venv/bin/python -m pip install .` in your existing checkout.

## Use

Open a new Codex session in your project and ask:

> Ask Claude to review my current diff. Use read-only access and give me the job ID.

After reconnecting, ask:

> Find my last Claude job, show me its review, and ask it which change to make first.

PatrickStar also supports Gemini through Antigravity. To see which agents are available, ask Codex to list the PatrickStar providers. For optional MathOS text consultations, fill in [the example config](examples/mathos-api.example.json), save it as `~/.config/patrickstar-agents/mathos-api.json`, and set its permissions to `600`.

Run the tests with `.venv/bin/python -m unittest discover -s tests`.
