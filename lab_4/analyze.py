import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
df = pd.read_json(HERE / "agent_log.jsonl", lines=True)
results = pd.DataFrame(json.loads((HERE / "results.json").read_text(encoding="utf-8")))

main_runs = results[~results.n.astype(str).str.startswith("B")].run_id
main = df[df.run_id.isin(main_runs)]
llm_calls = main[main.event.isin(["step", "final"])]
tokens = llm_calls.usage.apply(lambda u: u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))
tool_steps = main[main.event == "step"].dropna(subset=["action"])

print(f"Верных ответов: {results[results.auto_ok == True].shape[0]} из {results.auto_ok.notna().sum()} "
      f"(доля {results.auto_ok.dropna().astype(bool).mean():.0%})")
print(f"Среднее число шагов на вопрос: {llm_calls.groupby('run_id').size().mean():.2f}")
print(f"Средняя задержка LLM, с: {llm_calls.latency_s.mean():.2f}")
print(f"Доля успешных вызовов инструментов (tool_ok): {tool_steps.tool_ok.mean():.0%} "
      f"({int(tool_steps.tool_ok.sum())} из {len(tool_steps)})")
print(f"Суммарно токенов: {int(tokens.sum())} "
      f"(prompt {int(llm_calls.usage.apply(lambda u: u.get('prompt_tokens', 0)).sum())}, "
      f"completion {int(llm_calls.usage.apply(lambda u: u.get('completion_tokens', 0)).sum())})")

print("\nПо вопросам:")
print(results[["n", "type", "steps", "tokens", "auto_ok"]].to_string(index=False))
