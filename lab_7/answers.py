import json
import os
import sys
import time

from dotenv import load_dotenv

from rag import BASE, Pipeline, is_relevant, load_kb

load_dotenv(os.path.join(BASE, "..", ".env"))

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

SYSTEM = ("Ты ассистент сотрудников компании АлаТех. Отвечай ТОЛЬКО на основе контекста. "
          "Если ответа в контексте нет, напиши: «В документах нет информации». "
          "После ответа укажи doc_id источника в квадратных скобках. Отвечай кратко.")


def build_llm():
    if PROVIDER == "groq":
        from groq import Groq
        client = Groq()

        def call(messages):
            r = client.chat.completions.create(model=GROQ_MODEL, temperature=0, messages=messages)
            return r.choices[0].message.content

        return call, GROQ_MODEL

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, reasoning=False)

    def call(messages):
        return llm.invoke([(m["role"], m["content"]) for m in messages]).content

    return call, OLLAMA_MODEL


def parse_config(text):
    size, ov = text.split("/")
    return int(size), int(ov)


def main():
    config = parse_config(sys.argv[1]) if len(sys.argv) > 1 else (500, 75)
    call, model = build_llm()
    docs, questions = load_kb()
    pipe = Pipeline(docs, questions)
    pipe.add_config(*config)

    answers = []
    for q in questions:
        ctx = pipe.retrieve(config, q["question"])
        ctx_text = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in ctx)
        answer = call([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": f"Контекст:\n{ctx_text}\n\nВопрос: {q['question']}"}])
        found = any(is_relevant(c["text"], q) for c in ctx)
        answers.append({"id": q["id"], "вопрос": q["question"],
                        "найден ли нужный чанк": "да" if found else "нет",
                        "эталон": q["reference"], "ответ модели": answer.strip()})
        print(f"{q['id']} | чанк: {'да' if found else 'нет'}\n  эталон: {q['reference']}\n  ответ:  {answer.strip()}\n", flush=True)
        if PROVIDER == "groq":
            time.sleep(1)

    out = os.path.join(BASE, f"answers_{config[0]}_{config[1]}.json")
    json.dump({"config": list(config), "provider": PROVIDER, "model": model, "answers": answers},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("сохранено:", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
