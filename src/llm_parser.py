import os
import json
import urllib.request
import urllib.error
from typing import Tuple, Dict, Any

from core import (
    ParsedIntent,
    TimeWindow,
    ComfortPreference,
    CostPreference,
    DRPreference,
    parse_command_stub,
)


def _extract_json_object(text: str) -> dict:
    s = (text or "").strip()

    # Remove markdown code fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()

    # Direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])

    raise ValueError("No valid JSON object found in LLM response.")


def _clamp_hour(value, default):
    try:
        v = int(value)
        return max(0, min(23, v))
    except Exception:
        return default


def _clamp_hour_end(value, default):
    try:
        v = int(value)
        return max(1, min(24, v))
    except Exception:
        return default


def _clamp_float(value, default, lo=0.0, hi=1.0):
    try:
        v = float(value)
        return max(lo, min(hi, v))
    except Exception:
        return default


def _normalize_to_parsed_intent(command: str, data: dict) -> ParsedIntent:
    guest_event = bool(data.get("guest_event", False))
    guest_arrival_hour = data.get("guest_arrival_hour", 19 if guest_event else None)

    start_hour = _clamp_hour(data.get("time_window_start_hour", 18), 18)
    end_hour = _clamp_hour_end(data.get("time_window_end_hour", 23), 23)
    if end_hour <= start_hour:
        end_hour = min(24, start_hour + 1)

    thermal_label = str(data.get("thermal_label", "neutral")).lower()
    if thermal_label not in {"warm", "neutral", "cool"}:
        thermal_label = "neutral"

    comfort_priority = _clamp_float(data.get("comfort_priority", 0.6), 0.6)

    cost_label = str(data.get("cost_sensitivity_label", "medium")).lower()
    if cost_label not in {"low", "medium", "high"}:
        cost_label = "medium"

    cost_priority = _clamp_float(data.get("cost_priority", 0.5), 0.5)

    soft_budget = data.get("soft_budget_eur", None)
    try:
        soft_budget = None if soft_budget is None else float(soft_budget)
    except Exception:
        soft_budget = None

    dr_participation = str(data.get("dr_participation", "conditional")).lower()
    if dr_participation not in {"yes", "no", "conditional"}:
        dr_participation = "conditional"

    dr_priority = _clamp_float(data.get("dr_priority", 0.6), 0.6)
    clarification_needed = bool(data.get("clarification_needed", False))

    notes = data.get("notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]

    return ParsedIntent(
        raw_command=command,
        guest_event=guest_event,
        guest_arrival_hour=guest_arrival_hour if guest_arrival_hour is None else int(guest_arrival_hour),
        time_window=TimeWindow(start_hour=start_hour, end_hour=end_hour),
        comfort=ComfortPreference(
            thermal_label=thermal_label,
            comfort_priority=comfort_priority,
        ),
        cost=CostPreference(
            cost_sensitivity_label=cost_label,
            cost_priority=cost_priority,
            soft_budget_eur=soft_budget,
        ),
        dr=DRPreference(
            participation_willingness=dr_participation,
            dr_priority=dr_priority,
        ),
        clarification_needed=clarification_needed,
        notes=[str(n) for n in notes],
    )


def _openai_compatible_parse(command: str) -> ParsedIntent:
    """
    OpenAI-compatible endpoint parser.
    Defaults are set for Vector Engine.
    Required env vars:
      - OPENAI_API_KEY
      - LLM_MODEL
    Optional:
      - OPENAI_BASE_URL (default https://api.vectorengine.ai/v1)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.vectorengine.ai/v1").rstrip("/")

    if not api_key or not model:
        raise RuntimeError("Missing OPENAI_API_KEY and/or LLM_MODEL env vars.")

    system_prompt = (
        "You are an energy HEMS intent parser. "
        "Return ONLY a JSON object with these fields: "
        "{"
        "\"guest_event\": bool, "
        "\"guest_arrival_hour\": int|null, "
        "\"time_window_start_hour\": int, "
        "\"time_window_end_hour\": int, "
        "\"thermal_label\": \"warm|neutral|cool\", "
        "\"comfort_priority\": float, "
        "\"cost_sensitivity_label\": \"low|medium|high\", "
        "\"cost_priority\": float, "
        "\"soft_budget_eur\": number|null, "
        "\"dr_participation\": \"yes|no|conditional\", "
        "\"dr_priority\": float, "
        "\"clarification_needed\": bool, "
        "\"notes\": [string]"
        "} "
        "Use 24-hour local time assumptions. No markdown, no commentary."
    )

    user_prompt = f"Parse this command for smart-home energy scheduling:\n{command}"

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from LLM endpoint: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"Network/API call failed: {e}") from e

    data = json.loads(raw)

    # OpenAI-compatible chat completion shape
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:800]}") from e

    parsed_json = _extract_json_object(content)
    return _normalize_to_parsed_intent(command, parsed_json)


def parse_command_llm(command: str) -> Tuple[ParsedIntent, Dict[str, Any]]:
    """
    Parse via LLM if configured; otherwise fallback to stub parser.
    Returns: (intent, meta)
    """
    backend = os.getenv("LLM_BACKEND", "auto").lower()

    if backend in {"off", "stub", "none"}:
        intent = parse_command_stub(command)
        intent.notes.append("LLM parser disabled; used stub parser.")
        return intent, {"parser_used": "stub", "fallback": True, "error": None}

    try:
        intent = _openai_compatible_parse(command)
        return intent, {"parser_used": "llm", "fallback": False, "error": None}
    except Exception as e:
        intent = parse_command_stub(command)
        intent.notes.append(f"LLM fallback to stub: {type(e).__name__}")
        return intent, {"parser_used": "stub", "fallback": True, "error": str(e)}


def parse_command_by_mode(command: str, parser_mode: str):
    """
    parser_mode: 'stub' or 'llm'
    Returns (intent, parser_meta)
    """
    mode = (parser_mode or "stub").lower().strip()

    if mode == "stub":
        intent = parse_command_stub(command)
        return intent, {"parser_used": "stub", "fallback": False, "error": None}

    if mode == "llm":
        return parse_command_llm(command)

    raise ValueError(f"Unknown parser_mode: {parser_mode}. Use 'stub' or 'llm'.")