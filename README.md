# AI Agentic Labs

Лабораторные по курсу AI Agents for Enterprise Process Automation.

## Lab 1: Comparing LLMs (Groq, Gemini, OpenRouter)

Prompt used for every model: *"Explain the difference between machine learning and LLMs"* (sent in Russian).

| Platform | Model | Time (sec) | Tokens | Notes |
|---|---|---|---|---|
| Groq | `openai/gpt-oss-20b` | 2.85 | 1961 | Fastest, no failures |
| Gemini | `gemini-3.6-flash` | 14.52 | 2097 | Slowest, occasional 503 overload errors |
| OpenRouter | `openrouter/free` | 12.73 | 2752 | Longest answer |

![Model comparison](lab_1/model_comparison.png)

**Conclusion:** Groq was the fastest at 2.85 s, roughly 4.5–5 times quicker than OpenRouter and Gemini on the same prompt. Two model IDs from the course examples had already stopped working (`llama-3.1-8b-instant`, `gemini-2.5-flash`), so current replacements were used instead.

## Lab 2: Structured extraction (Pydantic + instructor)


**Entity:** Invoice.

```python
class Invoice(BaseModel):
    number: str
    amount: float
    currency: str
    issue_date: date
```

**Input (noisy on purpose):** amount spelled out in words, non-ISO date — "Накладная № НК-2025/104 от 3 марта 2025 г. Итого к оплате: тридцать две тысячи пятьсот сомов."

**Raw response (Groq, `openai/gpt-oss-20b`):** clean JSON on all 3 runs, no preamble/wrapper — the one-shot example in the prompt locked the format reliably.

**Typed response (`instructor`, no prompt needed):**
```
number='НК-2025/104' amount=32500.0 currency='сом' issue_date=datetime.date(2025, 3, 3)
```
Correctly parsed the spelled-out amount and non-ISO date.

**Comparison:** JSON syntax was never the problem here (`json.loads` and `Pydantic` both passed every run). The real issue was semantic: `currency='сом'` instead of `'KGS'` — `str` accepts anything, so it wasn't caught. Fix: `Literal["KGS", "USD", "EUR", "RUB"]`.

**What didn't work:** outdated model IDs (see Lab 1); silent `currency` bug caught only by inspection, not validation; notebook kept defaulting to system Python instead of project venv.

**AI tools used:** Claude for explaining instructor/Pydantic and debugging errors; all code run and verified locally.

## Setup

```bash
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and fill in your keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`).
