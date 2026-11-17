# PatrickStar Agents

Let Codex delegate work to Claude Code and Gemini through MCP. Each agent runs in the project folder you choose. You can close the MCP client, reconnect later, and continue the same agent conversation.

## Install

On macOS, install Codex CLI and either Claude Code (`claude`) or Antigravity CLI (`agy`). Sign in to the agent you want to use and clone the repository.

```sh
git clone https://github.com/guanyangwang/patrickstar-agents.git
cd patrickstar-agents
```

**Apple Silicon.** Use a native Python 3.11+ (`python3 -c 'import platform; print(platform.machine())'` should print `arm64`).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
codex mcp add patrickstar_agents -- "$PWD/.venv/bin/patrickstar-agents"
```

**Intel Mac or x86_64 Python under Rosetta.** Use an Intel (`osx-64`) conda installation, such as [Miniforge for macOS x86_64](https://github.com/conda-forge/miniforge#install). Run these commands from the repository. They create a separate environment even if a previous install left a partial `.venv`.

```sh
conda env create --prefix "$PWD/.conda-intel" --file environment-intel-macos.yml
codex mcp add patrickstar_agents -- "$PWD/.conda-intel/bin/patrickstar-agents"
```

The Intel environment gets a prebuilt `cryptography` package from conda-forge. [Upstream stopped publishing Intel macOS wheels on PyPI in version 49](https://cryptography.io/en/49.0.0/changelog/).

## Use

Open a new Codex session in your project and ask:

> Ask Claude to review my current diff. Use read-only access and give me the job ID.

After reconnecting, ask:

> Find my last Claude job, show me its review, and ask it which change to make first.

PatrickStar also supports Gemini through Antigravity. To see which agents are available, ask Codex to list the PatrickStar providers. For optional MathOS text consultations, fill in [the example config](examples/mathos-api.example.json), save it as `~/.config/patrickstar-agents/mathos-api.json`, and set its permissions to `600`.

Run the tests with `.venv/bin/python -m unittest discover -s tests` on Apple Silicon or `.conda-intel/bin/python -m unittest discover -s tests` in the Intel environment.
