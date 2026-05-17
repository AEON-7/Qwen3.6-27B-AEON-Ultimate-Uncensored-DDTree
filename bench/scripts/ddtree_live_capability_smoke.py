#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def post_chat(client: httpx.Client, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=None)
        elapsed = time.perf_counter() - started
        body_text = response.text
        parsed: Any
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        return {
            "ok": response.status_code == 200,
            "status": response.status_code,
            "elapsed_s": round(elapsed, 3),
            "json": parsed,
            "body_prefix": body_text[:500] if response.status_code != 200 else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "elapsed_s": None, "error": f"{type(exc).__name__}: {exc}"}


def choice_summary(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("json") if isinstance(result.get("json"), dict) else {}
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    tool_calls = message.get("tool_calls") or []
    usage = data.get("usage") or {}
    return {
        "content": content,
        "content_len": len(content) if isinstance(content, str) else None,
        "content_prefix": content[:240] if isinstance(content, str) else None,
        "reasoning_len": len(reasoning) if isinstance(reasoning, str) else None,
        "reasoning_prefix": reasoning[:240] if isinstance(reasoning, str) else None,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
    }


def classify(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    content = summary.get("content")
    text = content if isinstance(content, str) else ""
    out: dict[str, Any] = {}
    if name == "math_17x23":
        out["contains_391"] = "391" in text
        out["contains_known_bad_39_only"] = bool(re.fullmatch(r"\s*39\s*", text or ""))
    elif name == "guided_json":
        out["json_content_parseable"] = False
        if text:
            try:
                json.loads(text)
                out["json_content_parseable"] = True
            except Exception:
                pass
    elif name == "tool_call":
        out["has_openai_tool_call"] = summary.get("tool_call_count", 0) > 0
        out["raw_tool_text"] = "weather" in text.lower() or "tool" in text.lower()
    elif name == "prose":
        out["has_visible_content"] = bool(text.strip())
        out["reasoning_only"] = not bool(text.strip()) and bool(summary.get("reasoning_len"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="aeon-ultimate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    json_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature_f": {"type": "integer"},
            "condition": {"type": "string"},
        },
        "required": ["city", "temperature_f", "condition"],
        "additionalProperties": False,
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    tests: list[tuple[str, dict[str, Any]]] = [
        (
            "prose",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Write one vivid sentence about sunrise over the ocean. Do not explain."}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        (
            "math_17x23",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Compute 17 multiplied by 23. Return only the final integer."}],
                "max_tokens": 48,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        (
            "guided_json",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Return the weather for Dallas as JSON: city Dallas, temperature 72, condition clear."}],
                "max_tokens": 96,
                "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": {"name": "weather", "schema": json_schema}},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        (
            "tool_call",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Use the provided tool to get the weather in Dallas."}],
                "max_tokens": 96,
                "temperature": 0,
                "tools": tools,
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
    ]

    with httpx.Client(timeout=None) as client:
        model_info = client.get(f"{args.base_url.rstrip('/')}/models", timeout=10).json()
        rows = []
        for name, payload in tests:
            result = post_chat(client, args.base_url, payload)
            summary = choice_summary(result)
            rows.append(
                {
                    "name": name,
                    "payload_summary": {
                        "model": payload.get("model"),
                        "max_tokens": payload.get("max_tokens"),
                        "temperature": payload.get("temperature"),
                        "has_response_format": "response_format" in payload,
                        "has_tools": "tools" in payload,
                    },
                    "result": result,
                    **summary,
                    "checks": classify(name, summary),
                }
            )
            print(f"{name}: ok={result.get('ok')} status={result.get('status')} finish={summary.get('finish_reason')} checks={rows[-1]['checks']}")

    artifact = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "model_info": model_info,
        "tests": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(artifact, indent=2))
    print(f"saved={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
