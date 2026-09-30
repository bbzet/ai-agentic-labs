# Lab 1: Comparing LLMs (Groq, Gemini, OpenRouter)

Prompt used for every model: *"Explain the difference between machine learning and LLMs"* (sent in Russian).

## Results

| Platform | Model | Time (sec) | Tokens | Notes |
|---|---|---|---|---|
| Groq | `openai/gpt-oss-20b` | 2.85 | 1961 | Fastest, no failures |
| Gemini | `gemini-3.6-flash` | 14.52 | 2097 | Slowest, occasional 503 overload errors |
| OpenRouter | `openrouter/free` | 12.73 | 2752 | Longest answer |

![Model comparison](model_comparison.png)

## Conclusion

Groq was the fastest at 2.85 s, roughly 4.5–5 times quicker than OpenRouter and Gemini on the same prompt. The models also differed in verbosity: answers ranged from 1961 to 2752 tokens for an identical request. Two model IDs from the course examples had already stopped working: Groq retired `llama-3.1-8b-instant`, and Google closed `gemini-2.5-flash` to new users, so current replacements were used instead. For further work Groq is the pick — lowest latency, no failures — while Gemini intermittently returned 503 errors caused by provider-side overload.