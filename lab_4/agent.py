import json
import os
import re
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from tools import calculator, text_search

LOG_PATH = Path(__file__).parent / "agent_log.jsonl"
TOOLS = {"calculator": calculator, "text_search": text_search}

SYSTEM_PROMPT = """Ты — ассистент, который решает задачи пошагово в формате ReAct.
Доступные инструменты:
{tools}

Строго соблюдай формат. На каждом шаге пиши:
Thought: рассуждение о том, что делать дальше
Action: имя_инструмента
Action Input: входные данные инструмента одной строкой

После этого ОСТАНОВИСЬ и жди строку Observation с результатом.
Когда информации достаточно, напиши:
Thought: итоговое рассуждение
Final Answer: ответ пользователю

Правила: не придумывай Observation сам; используй только перечисленные инструменты;
все вычисления делай через calculator; факты о компании бери только из text_search.
Если поиск ничего не нашёл, переформулируй запрос (одно-два ключевых слова). Если после
нескольких попыток ответа в документах нет, так и скажи в Final Answer, ничего не выдумывай."""

TOOLS_DESC = ("- calculator: вычисляет арифметическое выражение (числа и + - * / % **). "
              "Вход: выражение, например 1500 * 4\n"
              "- text_search: ищет фрагменты во внутренних документах компании. "
              "Вход: короткий запрос из ключевых слов, например суточные командировки")

ACTION_RE = re.compile(r"Action:\s*([\w\-]+)\s*\n\s*Action Input:\s*(.+)")
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)

LLMReply = namedtuple("LLMReply", "text usage")


def parse(text: str):
    text = text.split("Observation:")[0]
    m_act, m_fin = ACTION_RE.search(text), FINAL_RE.search(text)
    if m_fin and (not m_act or m_fin.start() < m_act.start()):
        return ("final", m_fin.group(1).strip())
    if m_act:
        return ("action", m_act.group(1).strip(), m_act.group(2).strip())
    return ("error", "Нет 'Action:'+'Action Input:' и нет 'Final Answer:'. Соблюдай формат.")


def call_llm(messages) -> LLMReply:
    from dotenv import load_dotenv
    from openai import BadRequestError, OpenAI, RateLimitError
    load_dotenv(Path(__file__).parent.parent / ".env")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("GROQ_API_KEY") if "groq" in base_url else os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    client = OpenAI(base_url=base_url, api_key=api_key)

    kwargs = dict(model=model, messages=messages, temperature=0, stop=["Observation:"], max_tokens=400)
    if "groq" not in base_url:
        kwargs["reasoning_effort"] = "none"
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except BadRequestError:
            if "stop" not in kwargs:
                raise
            kwargs.pop("stop")
        except RateLimitError:
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("LLM недоступна: исчерпаны повторные попытки")
    u = resp.usage
    return LLMReply((resp.choices[0].message.content or "").strip(),
                    {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})


def make_mock_llm(script):
    replies = iter(script)
    return lambda messages: next(replies)


def _log(path, run_id, **record):
    record = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "run_id": run_id, **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_agent(question: str, max_steps: int = 8, llm=call_llm, log_path=LOG_PATH) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(tools=TOOLS_DESC)},
                {"role": "user", "content": f"Question: {question}"}]
    _log(log_path, run_id, step=0, event="start", question=question, max_steps=max_steps)

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        reply = llm(messages)
        latency = round(time.perf_counter() - t0, 3)
        text, usage = (reply, {}) if isinstance(reply, str) else reply
        parsed = parse(text)
        base = dict(step=step, llm_output=text, latency_s=latency, usage=usage)

        if parsed[0] == "final":
            _log(log_path, run_id, event="final", answer=parsed[1], **base)
            return parsed[1]

        if parsed[0] == "action":
            _, name, arg = parsed
            if name in TOOLS:
                try:
                    observation = str(TOOLS[name](arg))
                except Exception as e:
                    observation = f"Ошибка инструмента {name}: {e}"
                tool_ok = not observation.startswith("Ошибка")
            else:
                observation = f"Ошибка: неизвестный инструмент '{name}'. Доступные: {', '.join(TOOLS)}."
                tool_ok = False
            _log(log_path, run_id, event="step", action=name, action_input=arg,
                 observation=observation, tool_ok=tool_ok, **base)
        else:
            observation = parsed[1]
            _log(log_path, run_id, event="step", action=None, action_input=None,
                 observation=observation, tool_ok=False, **base)

        messages.append({"role": "assistant", "content": text.split("Observation:")[0].rstrip()})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    _log(log_path, run_id, step=max_steps, event="max_steps_reached")
    return "Не удалось получить ответ за отведённое число шагов."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Ваш вопрос: ")
    print(run_agent(q))
