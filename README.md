# Lab 1: Comparing LLMs (Groq, Gemini, OpenRouter)

Prompt used for every model: *"Explain the difference between machine learning and LLMs"* (sent in Russian).

Each model was called programmatically through its Python SDK. Response time and total token count were taken from the API response.

## Results

| Platform | Model | Time (sec) | Tokens | Notes |
|---|---|---|---|---|
| Groq | `openai/gpt-oss-20b` | 2.85 | 1961 | Fastest, no failures |
| Gemini | `gemini-3.6-flash` | 14.52 | 2097 | Slowest, occasional 503 overload errors |
| OpenRouter | `openrouter/free` | 12.73 | 2752 | Longest answer |

![Model comparison](model_comparison.png)

## Conclusion

Groq was the fastest at 2.85 s, roughly 4.5 to 5 times quicker than OpenRouter (12.73 s) and Gemini (14.52 s) on the same prompt. The models also differed in verbosity: answers ranged from 1961 to 2752 tokens for an identical request, with OpenRouter being the most verbose. Timings varied between runs (for example, OpenRouter took 25.18 s in an earlier run), presumably because the free router does not always serve the same model. One unexpected finding was that two model IDs used in the course examples had already stopped working: Groq retired `llama-3.1-8b-instant`, and Google closed `gemini-2.5-flash` to new users, so I had to switch to current replacements. For further work I would pick Groq because of its low latency and stable responses, whereas Gemini intermittently returned 503 errors caused by provider-side overload.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`). The `.env` file is git-ignored. Then run the cells in `lab_1.ipynb` from top to bottom.
