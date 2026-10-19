import json
import os
import sys
import time

os.environ.setdefault("LLM_PROVIDER", "ollama")

import hr_agent as hr


def traced(classifier, calls):
    def wrapper(text):
        started = time.time()
        try:
            req_type, confidence = classifier(text)
        except Exception as exc:
            calls.append({"text": text, "error": type(exc).__name__})
            raise
        calls.append({
            "text": text,
            "req_type": req_type,
            "confidence": round(confidence, 2),
            "latency_s": round(time.time() - started, 1),
        })
        return req_type, confidence

    return wrapper


def run_mode(name, classifier):
    results = []
    for tid, text, replies, expected in hr.TESTS:
        calls = []
        hr.CLASSIFY = traced(classifier, calls)
        graph = hr.build_graph()
        res, status = hr.run_case(graph, f"{name}-{tid}", text, replies)
        problems = hr.check_case(tid, res, status, expected)
        results.append({
            "id": tid,
            "request": text,
            "replies": replies,
            "expected": expected,
            "route": res["route_log"],
            "final": hr.final_node(res["route_log"]),
            "passed": not problems,
            "problems": problems,
            "classifier_calls": calls,
        })
        print(name, tid, "PASS" if not problems else "FAIL", " -> ".join(res["route_log"]), flush=True)
    return results


if __name__ == "__main__":
    modes = {"keywords": hr.keyword_classify, "llm": hr.llm_classify}
    selected = sys.argv[1:] or list(modes)
    out = {}
    for name in selected:
        out[name] = run_mode(name, modes[name])
        print(name, sum(r["passed"] for r in out[name]), "/", len(out[name]), flush=True)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"llm_model": os.getenv("OLLAMA_MODEL", "qwen3.5:9b"), **out}, f, ensure_ascii=False, indent=2)
