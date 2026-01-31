from __future__ import annotations

import json
import os
import urllib.request


def call_openai_chat(prompt: str, model: str, temperature: float, max_tokens: int, api_base: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise experiment analyst. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI response missing choices")
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if not content:
        raise RuntimeError("OpenAI response missing content")
    return content


def call_ollama_chat(prompt: str, model: str, temperature: float, api_base: str) -> str:
    url = api_base.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise experiment analyst. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    message = data.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("Ollama response missing content")
    return content
