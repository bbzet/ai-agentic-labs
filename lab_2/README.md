# Lab 2: Structured extraction (Pydantic + instructor)

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