from langgraph.types import Command

import hr_agent as hr


def show_mermaid(graph):
    print("=== draw_mermaid() ===")
    print(graph.get_graph().draw_mermaid())


def show_stream(graph):
    print("=== stream_mode='updates': T08 ===")
    cfg = {"configurable": {"thread_id": "dbg-T08"}}
    inputs = {"request": "Хочу отпуск и справку о зарплате", "route_log": []}
    for chunk in graph.stream(inputs, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"{node}: {update}")
    print("next во время паузы:", graph.get_state(cfg).next)
    for chunk in graph.stream(Command(resume="справка"), cfg, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"{node}: {update}")
    print("next после завершения:", graph.get_state(cfg).next)
    print()


def show_history(graph):
    print("=== get_state_history: T06 ===")
    cfg = {"configurable": {"thread_id": "dbg-T06"}}
    res = graph.invoke({"request": "Помогите", "route_log": []}, cfg)
    for reply in ("не знаю", "просто так"):
        res = graph.invoke(Command(resume=reply), cfg)
    history = list(graph.get_state_history(cfg))
    print(f"снимков: {len(history)}")
    for snap in reversed(history):
        meta = snap.metadata
        print(
            f"step={meta['step']:>2} source={meta['source']:<5} next={str(snap.next):<22} "
            f"clarify_count={snap.values.get('clarify_count', '-')!s:<2} "
            f"route_log={snap.values.get('route_log', [])}"
        )
    print()


if __name__ == "__main__":
    hr.CLASSIFY = hr.keyword_classify
    g = hr.build_graph()
    show_mermaid(g)
    show_stream(g)
    show_history(g)
