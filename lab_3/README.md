## Lab 3: Open models and MCP (Ollama + MCP server + agent)

A small local agent: it answers questions about company documents (`docs/`) and decides on its own when to search them. Everything runs on the laptop, no API keys needed.

**Structure**
```
lab_3/
├── docs/               # fictional company regulations (hr_vacation.txt, it_faq.txt)
├── server.py           # MCP server with the search_files tool
├── check.py            # tests the MCP server without a model
└── agent.py            # agent: Ollama model + MCP tool-calling loop
```

**Run**
```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install "mcp[cli]>=2,<3" ollama
ollama pull qwen3.5:9b
python check.py                  # test the MCP server
python agent.py                  # ask the agent a question
```

### Step 1. Model

| Model | eval rate | Memory (`ollama ps`) | Processor |
|---|---|---|---|
| `qwen3.5:9b` | 5.33 tokens/s | 6.2 GB | 100% CPU |

The model ran on CPU only, which explains the low speed. Prompt eval rate was 14.18 tokens/s.

### Step 2. MCP server

`python check.py` finds the `search_files` tool and returns all six lines of `hr_vacation.txt` containing "отпуск".

1. **`@mcp.tool()`** registers the function as an MCP tool: the server publishes its name, description and argument schema (built from the type hints), so a client can list it with `list_tools()` and call it with `call_tool()`.
2. **The docstring** is the tool description. It is read by the model (through the agent), not by a human: it is how the model decides whether and when to call the tool, so it must say what the tool does and when to use it.
3. **"пароль" instead of "отпуск"** matches only `it_faq.txt` ("Пароль от рабочей почты меняется каждые 90 дней."), because the search is a case-insensitive substring match over all files in `docs/`.

### Step 3. Agent

The line replacing the `TODO` in `agent.py`:
```python
result = await client.call_tool(call.function.name, call.function.arguments)
```

Question "Сколько дней отпуска положено сотруднику?":
```
Шаг 1: search_files({'query': 'отпуск дней положено сотрудник'})
Шаг 2: search_files({'query': 'отпуск сотруднику'})
Шаг 3: search_files({'query': 'отпуск'})
Согласно документу hr_vacation.txt, ежегодный оплачиваемый отпуск составляет 28 календарных дней.
```
Question "Какая зарплата у директора?" (answer is not in the documents):
```
Шаг 1: search_files({'query': 'зарплата директора директор оклад'})
К сожалению, в документах компании нет информации о зарплате директора.
```
The model searched, found nothing and said so instead of inventing a number.

Observation: the first two searches returned nothing because the model packed several words into one query, and `search_files` matches the whole phrase as a substring. Only the single-word query "отпуск" succeeded. The model retried on its own, which is the point of the agent loop.

### What didn't work and how it was solved

`StdioServerParameters(command="python", ...)` starts whatever `python` is first in `PATH`. Outside the activated venv that interpreter has no `mcp` installed, so the server subprocess fails to start. Fixed by activating the venv (or putting `venv/Scripts` first in `PATH`) before running `check.py` and `agent.py`.

Step 4 (comparing two models) and the bonus task were not done.

**AI tools used:** Claude Code for setting up the environment, running the scripts and writing this README; all code was run and verified locally.
