import json
import sys
import tempfile
from pathlib import Path

import agent
from agent import make_mock_llm, parse, run_agent
from tools import calculator, text_search

HERE = Path(__file__).parent


def _norm(s: str) -> str:
    return "".join(ch for ch in s if ch not in "   ,")


QUESTIONS = [
    {"n": 1, "type": "Поиск", "q": "Сколько календарных дней составляет ежегодный оплачиваемый отпуск?",
     "expected": "28 календарных дней", "must": ["28"]},
    {"n": 2, "type": "Поиск", "q": "Как часто нужно менять пароль от рабочей почты?",
     "expected": "каждые 90 дней", "must": ["90"]},
    {"n": 3, "type": "Вычисление", "q": "Посчитай значение выражения (12500 * 12) / 4.",
     "expected": "37500", "must": ["37500"]},
    {"n": 4, "type": "Поиск + вычисление",
     "q": "Годовой оклад сотрудника 600000 сом, оценка эффективности «превосходит ожидания». "
          "Какой ежегодный бонус он получит?",
     "expected": "120000 сом (20% от 600000)", "must": ["120000"]},
    {"n": 5, "type": "Поиск + вычисление",
     "q": "Командировка по стране длится 5 дней, в гостинице 4 ночи. Какова максимально возможная "
          "сумма суточных и проживания вместе?",
     "expected": "25500 сом (1500*5 + 4500*4)", "must": ["25500"]},
]

BREAK_QUESTIONS = [
    {"n": "B1", "type": "Ответа нет в документах", "q": "Какая зарплата у директора компании?",
     "expected": "в документах нет информации", "must": []},
    {"n": "B2", "type": "Несуществующий инструмент",
     "q": "Какая сейчас погода в Бишкеке? Обязательно используй инструмент weather.",
     "expected": "агент не должен получить погоду; корректно сообщить об ограничении", "must": []},
]


def test_mock():
    assert parse("Thought: x\nAction: text_search\nAction Input: суточные")[0:2] == ("action", "text_search")
    assert parse("Thought: x\nFinal Answer: 42") == ("final", "42")
    assert parse("Просто болтаю")[0] == "error"
    assert parse("Thought: x\nAction: calculator\nAction Input: 2+2\nObservation: 5\nFinal Answer: 5")[0] == "action"
    assert parse("Final Answer: готово\nAction: calculator\nAction Input: 1")[0] == "final"

    assert calculator("1500 * 4") == "6000"
    assert calculator("2 ^ 10") == "1024"
    assert calculator("1/0") == "Ошибка: деление на ноль"
    assert calculator("9**9**9").startswith("Ошибка")
    assert calculator("__import__('os').system('echo hacked')").startswith("Ошибка")
    assert "[hr_policy.txt" in text_search("суточные командировки")
    assert "Ничего не найдено" in text_search("квантовая телепортация")
    assert "[hr_policy.txt" in text_search("командировок")

    log = Path(tempfile.mkdtemp()) / "mock.jsonl"
    mock = make_mock_llm([
        "Thought: Нужна ставка суточных.\nAction: text_search\nAction Input: суточные",
        "Thought: Попробую несуществующий инструмент.\nAction: weather\nAction Input: Бишкек",
        "Я просто отвечу без формата",
        "Thought: Посчитаю.\nAction: calculator\nAction Input: 1500 * 4",
        "Thought: Готово.\nFinal Answer: 6000 сом",
    ])
    answer = run_agent("Сколько суточных за 4 дня?", llm=mock, log_path=log)
    assert answer == "6000 сом", answer
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["start", "step", "step", "step", "step", "final"]
    assert rows[1]["tool_ok"] is True
    assert rows[2]["tool_ok"] is False and "неизвестный инструмент" in rows[2]["observation"]
    assert rows[3]["action"] is None and "формат" in rows[3]["observation"]
    assert rows[4]["observation"] == "6000"

    loop = make_mock_llm(["Thought: ищу\nAction: text_search\nAction Input: суточные"] * 3)
    log2 = Path(tempfile.mkdtemp()) / "loop.jsonl"
    assert "Не удалось" in run_agent("?", max_steps=3, llm=loop, log_path=log2)
    last = json.loads(log2.read_text(encoding="utf-8").splitlines()[-1])
    assert last["event"] == "max_steps_reached"
    print("mock: все проверки пройдены")


def test_live():
    log = agent.LOG_PATH
    log.unlink(missing_ok=True)
    results = []
    for item in QUESTIONS + BREAK_QUESTIONS:
        print(f"\n=== {item['n']} [{item['type']}] {item['q']}")
        before = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
        answer = run_agent(item["q"])
        lines = log.read_text(encoding="utf-8").splitlines()[before:]
        rows = [json.loads(line) for line in lines]
        calls = [r for r in rows if r["event"] in ("step", "final")]
        tokens = sum(r["usage"].get("prompt_tokens", 0) + r["usage"].get("completion_tokens", 0) for r in calls)
        ok = bool(item["must"]) and all(m in _norm(answer) for m in item["must"])
        results.append({"n": item["n"], "type": item["type"], "question": item["q"],
                        "expected": item["expected"], "answer": answer, "steps": len(calls),
                        "tokens": tokens, "auto_ok": ok if item["must"] else None,
                        "run_id": rows[0]["run_id"]})
        print(f"-> {answer}\n   шагов: {len(calls)}, токенов: {tokens}, авто-проверка: {results[-1]['auto_ok']}")
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nРезультаты сохранены в results.json, журнал - в agent_log.jsonl")


if __name__ == "__main__":
    {"mock": test_mock, "live": test_live}[sys.argv[1] if len(sys.argv) > 1 else "mock"]()
