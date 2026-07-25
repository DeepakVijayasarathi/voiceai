import json
import statistics
import time

import httpx

BASE_URL = "http://127.0.0.1:8700"

TEST_SENTENCES = {
    "hin": "नमस्ते, आज मौसम बहुत अच्छा है।",
    "tam": "வணக்கம், இன்று வானிலை மிகவும் அழகாக உள்ளது.",
    "tel": "నమస్కారం, ఈ రోజు వాతావరణం చాలా బాగుంది.",
    "mal": "നമസ്കാരം, ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്.",
    "kan": "ನಮಸ್ಕಾರ, ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
    "mar": "नमस्कार, आज हवामान खूप छान आहे.",
    "guj": "નમસ્તે, આજે હવામાન ખૂબ સરસ છે.",
    "ben": "নমস্কার, আজ আবহাওয়া খুব সুন্দর।",
    "pan": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ।",
    "ory": "ନମସ୍କାର, ଆଜି ପାଣିପାଗ ବହୁତ ଭଲ ଅଛି।",
    "asm": "নমস্কাৰ, আজি বতৰ বহুত ভাল।",
}

REPEATS = 3
results = {}

with httpx.Client(timeout=60.0) as client:
    for lang, text in TEST_SENTENCES.items():
        times = []
        durations = []
        wav_path = f"/root/bench_{lang}.wav"
        for i in range(REPEATS):
            t0 = time.time()
            resp = client.post(f"{BASE_URL}/tts", json={"text": text, "language": lang})
            elapsed = time.time() - t0
            if resp.status_code != 200:
                print(f"{lang}: FAILED run {i}: {resp.status_code} {resp.text}", flush=True)
                continue
            times.append(elapsed)
            if i == REPEATS - 1:
                with open(wav_path, "wb") as f:
                    f.write(resp.content)

        if times:
            # First call includes lazy nothing (model already warm from
            # startup) so all calls are steady-state; report min/mean/p95.
            results[lang] = {
                "min_s": round(min(times), 3),
                "mean_s": round(statistics.mean(times), 3),
                "max_s": round(max(times), 3),
            }
            print(f"{lang}: min={results[lang]['min_s']}s mean={results[lang]['mean_s']}s max={results[lang]['max_s']}s -> {wav_path}", flush=True)

with open("/root/indic-tts-fast/benchmark/latency_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE")
