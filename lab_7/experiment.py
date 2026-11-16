import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rag import BASE, Pipeline, is_relevant, kb_stats, load_kb

CONFIGS = [(200, 30), (500, 75), (1000, 150), (500, 0), (500, 150)]
INSPECT = [("q03", (500, 75)), ("q09", (500, 75)), ("q03", (200, 30)), ("q09", (200, 30)),
           ("q03", (1000, 150)), ("q09", (1000, 150))]


def show_top5(pipe, qid, config):
    q = next(x for x in pipe.questions if x["id"] == qid)
    print(f"\n{q['id']} {config}: {q['question']}\nНужный документ: {q['gold_doc_id']}")
    for i, c in enumerate(pipe.retrieve(config, q["question"]), 1):
        mark = "V" if is_relevant(c["text"], q) else " "
        print(f"  {mark} {i}. {c['doc_id']:12s} dist={c['distance']:.3f}  {c['text'][:90]!r}")


def per_question(pipe):
    rows = []
    for config in CONFIGS:
        for q in pipe.questions:
            ranks = pipe.ranks(config, q)
            rows.append({"config": f"{config[0]}/{config[1]}", "id": q["id"],
                         "rank": ranks[0] + 1 if ranks else None})
    return pd.DataFrame(rows).pivot(index="id", columns="config", values="rank")


def main():
    docs, questions = load_kb()
    kb_stats(docs, questions)
    pipe = Pipeline(docs, questions)

    rows = []
    for size, ov in CONFIGS:
        chunks = pipe.add_config(size, ov)
        rows.append({"chunk_size": size, "overlap": ov, "чанков": len(chunks), **pipe.evaluate((size, ov))})
        print(f"готово: chunk_size={size}, overlap={ov}")

    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False))

    ranks = per_question(pipe)
    print("\nПозиция правильного чанка (пусто = не найден в top-5):")
    print(ranks.to_string())

    for qid, config in INSPECT:
        show_top5(pipe, qid, config)

    results.to_json(os.path.join(BASE, "experiment.json"), orient="records", force_ascii=False, indent=1)
    ranks.to_json(os.path.join(BASE, "ranks.json"), force_ascii=False, indent=1)

    main_rows = results.iloc[:3]
    ax = main_rows.plot.bar(x="chunk_size", y=["hit@1", "hit@5", "MRR"], rot=0, figsize=(8, 4),
                            title="Качество поиска при разных chunk_size")
    ax.set_ylim(0, 1)
    ax.set_ylabel("доля / MRR")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "retrieval_metrics.png"), dpi=130)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
