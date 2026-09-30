## Lab 2: Structured extraction (Pydantic + instructor)

**Entity:** Restaurant table reservation request.

```python
class ReservationRequest(BaseModel):
    customer_name: str
    party_size: int
    reservation_date: date
    reservation_time: str = Field(description="Время в формате HH:MM, 24-часовой формат")
    special_request: str | None = None

    @field_validator("reservation_time")
    @classmethod
    def check_time_format(cls, v: str) -> str:
        time.fromisoformat(v if len(v) > 5 else v + ":00")
        return v
```

**Input (noisy on purpose):** relative date without year, informal time, an optional detail folded into the same sentence — "Здравствуйте! Хотели бы столик на четверых в субботу, 8 ноября, часам к 7 вечера. Меня зовут Нурлан. Если можно, у окна — у нас будет день рождения."

**Prompt:**
```python
prompt = f'''Извлеки данные из текста заявки на бронирование столика в формате JSON.

Текст: "Добрый вечер, нужен стол на двоих завтра в 20:00, меня зовут Айгерим."
Результат: {{"customer_name": "Айгерим", "party_size": 2, "reservation_date": "2026-10-01", "reservation_time": "20:00", "special_request": null}}

Текст: "{raw_text}"
Результат:'''
```

**Typed response (`instructor`):**
```
customer_name='Нурлан' party_size=4 reservation_date=datetime.date(2026, 11, 8) reservation_time='19:00' special_request='у окна'
```
Correctly resolved "часам к 7 вечера" to 24-hour `19:00`, inferred the year for the date, and extracted only the actionable part of the special request ("у окна"), dropping the unrelated birthday mention.

**What didn't work and how it was solved:** the first model used `reservation_time: time` (Python's `datetime.time`). Pydantic maps that to JSON Schema's `format: "time"`, which requires RFC 3339 formatting (`HH:MM:SS`, minimum 9 characters) — but the model naturally wrote `"19:00"`. Groq validates tool-call arguments against the schema on its own side before `instructor` ever sees the response, so this came back as a hard `400 tool_use_failed` on the very first attempt (`instructor`'s `max_retries` never even kicked in). Fixed by changing the field to `str` with a custom `field_validator` that checks the format manually instead of relying on the schema's strict time type.

**AI tools used:** Claude for explaining instructor/Pydantic internals and diagnosing the Groq schema-validation error; all code run and verified locally.