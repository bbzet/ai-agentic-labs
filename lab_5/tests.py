import json
import sys

import httpx
from groq import APIConnectionError, AuthenticationError
from langchain_core.exceptions import OutputParserException
from langgraph.types import Command
from ollama import ResponseError
from pydantic import ValidationError

import hr_agent as hr


class Scripted:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        item = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(item, Exception):
            raise item
        return item


def use(classifier):
    hr.CLASSIFY = classifier
    return hr.build_graph()


def drive(graph, text, replies=(), tid="t"):
    return hr.run_case(graph, tid, text, replies)


def test_keyword_suite():
    hr.CLASSIFY = hr.keyword_classify
    g = hr.build_graph()
    for tid, text, replies, expected in hr.TESTS:
        res, status = hr.run_case(g, tid, text, replies)
        assert not hr.check_case(tid, res, status, expected), (tid, res["route_log"])


def test_empty_and_short_requests_go_to_fallback():
    scripted = Scripted([("vacation", 0.9)])
    g = use(scripted)
    for text in ("", "  ", "ab", None):
        res, _ = drive(g, text, tid=f"e-{text}")
        assert res["route_log"] == ["intake", "escalate", "respond"], res["route_log"]
        assert res["error"] == "empty_request"
        assert res["status"] == "escalated"
    assert scripted.calls == []
    res, _ = drive(g, "abc", tid="e-abc")
    assert res["route_log"][1] == "classify"


def test_threshold_boundary():
    g = use(Scripted([("vacation", 0.7)]))
    res, _ = drive(g, "запрос про отпуск", tid="b1")
    assert hr.final_node(res["route_log"]) == "handle_vacation"
    g = use(Scripted([("vacation", 0.69), ("vacation", 0.69), ("vacation", 0.69)]))
    res, status = drive(g, "запрос про отпуск", ["ещё", "ещё"], tid="b2")
    assert res["route_log"].count("clarify") == 2
    assert hr.final_node(res["route_log"]) == "escalate"


def test_unknown_type_with_high_confidence_is_not_routed():
    g = use(Scripted([("unknown", 0.99)]))
    res, _ = drive(g, "что-то странное", ["не знаю", "не знаю"], tid="u1")
    assert hr.final_node(res["route_log"]) == "escalate"


def test_max_clarify_limit():
    scripted = Scripted([("unknown", 0.0)])
    g = use(scripted)
    res, status = drive(g, "Помогите", ["a", "b", "c"], tid="m1")
    assert status == "DONE"
    assert res["clarify_count"] == hr.MAX_CLARIFY == 2
    assert res["route_log"] == [
        "intake", "classify", "clarify", "classify", "clarify", "classify", "escalate", "respond"
    ]
    assert len(scripted.calls) == 3
    assert res["escalation_reason"] == "unclear_type"


def test_classifier_sees_only_latest_clarification():
    scripted = Scripted([("vacation", 0.4), ("certificate", 0.9)])
    g = use(scripted)
    res, _ = drive(g, "Хочу отпуск и справку о зарплате", ["справка"], tid="c1")
    assert scripted.calls == ["Хочу отпуск и справку о зарплате", "справка"]
    assert hr.final_node(res["route_log"]) == "handle_certificate"


def test_interrupt_payload_is_json_and_graph_waits():
    g = use(Scripted([("unknown", 0.0)]))
    cfg = {"configurable": {"thread_id": "p1"}}
    res = g.invoke({"request": "У меня вопрос", "route_log": []}, cfg)
    payload = res["__interrupt__"][0].value
    json.dumps(payload)
    assert payload["attempt"] == 1 and payload["max_attempts"] == 2
    assert g.get_state(cfg).next == ("clarify",)
    res = g.invoke(Command(resume="не знаю"), cfg)
    assert res["__interrupt__"][0].value["attempt"] == 2


def test_threads_are_isolated():
    g = use(Scripted([("unknown", 0.0)]))
    a = g.invoke({"request": "вопрос A", "route_log": []}, {"configurable": {"thread_id": "A"}})
    b = g.invoke({"request": "вопрос B", "route_log": []}, {"configurable": {"thread_id": "B"}})
    assert a["request"] == "вопрос A" and b["request"] == "вопрос B"
    assert a["route_log"] == ["intake", "classify", "clarify"] or "__interrupt__" in a


def _validation_error():
    try:
        hr.Classification(req_type="vacation", confidence=5)
    except ValidationError as exc:
        return exc


def test_classifier_errors_lead_to_fallback():
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(401, request=request)
    errors = [
        TimeoutError("timeout"),
        ConnectionError("refused"),
        ValueError("bad json"),
        _validation_error(),
        OutputParserException("bad output"),
        ResponseError("model not found"),
        httpx.ConnectError("down", request=request),
        APIConnectionError(request=request),
        AuthenticationError("401", response=response, body=None),
    ]
    for i, exc in enumerate(errors):
        g = use(Scripted([exc]))
        res, status = drive(g, "Хочу отпуск", tid=f"err-{i}")
        assert status == "DONE", type(exc)
        assert res["route_log"] == ["intake", "classify", "escalate", "respond"], (type(exc), res["route_log"])
        assert res["error"] == "classifier_error"
        assert res["error_detail"].startswith(type(exc).__name__)
        assert "HR-специалисту" in res["answer"]


def test_unexpected_exceptions_are_not_swallowed():
    g = use(Scripted([RuntimeError("bug in code")]))
    try:
        drive(g, "Хочу отпуск", tid="bug")
    except RuntimeError:
        return
    raise AssertionError("RuntimeError must propagate")


def test_error_after_clarification_goes_to_fallback():
    g = use(Scripted([("unknown", 0.0), TimeoutError("late")]))
    res, _ = drive(g, "У меня вопрос", ["про отпуск"], tid="late")
    assert res["route_log"] == ["intake", "classify", "clarify", "classify", "escalate", "respond"]
    assert res["error"] == "classifier_error"


def test_answers_never_approve():
    hr.CLASSIFY = hr.keyword_classify
    g = hr.build_graph()
    for i, (kind, _) in enumerate(hr.HR_POLICY.items()):
        res, _ = drive(g, f"Одобри мне {kind} {hr.KEYWORDS[kind][0]}", tid=f"na-{i}")
        assert not hr.FORBIDDEN_WORDS.search(res["answer"]), res["answer"]
    out = hr.respond({"answer": "Ваш отпуск согласован и одобрен."})
    assert out["answer"] == hr.SAFE_ANSWER


def test_route_log_has_one_entry_per_node_visit():
    hr.CLASSIFY = hr.keyword_classify
    g = hr.build_graph()
    res, _ = hr.run_case(g, "log", "Хочу отпуск и справку о зарплате", ["справка"])
    assert res["route_log"] == ["intake", "classify", "clarify", "classify", "handle_certificate", "respond"]


def test_resume_without_checkpointer_is_impossible():
    raw = hr.StateGraph(hr.HRState)
    raw.add_node("clarify", hr.clarify)
    raw.add_edge(hr.START, "clarify")
    raw.add_edge("clarify", hr.END)
    graph = raw.compile()
    res = graph.invoke({"request": "x", "route_log": [], "clarify_count": 0})
    assert "__interrupt__" in res
    try:
        graph.invoke(Command(resume="про отпуск"))
    except RuntimeError as exc:
        assert "checkpointer" in str(exc)
        return
    raise AssertionError("resume without checkpointer must fail")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    original = hr.CLASSIFY
    failed = 0
    for fn in TESTS:
        hr.CLASSIFY = original
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as exc:
            failed += 1
            print("FAIL", fn.__name__, type(exc).__name__, exc)
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
