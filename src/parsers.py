"""
Parsing strategies for natural-language HEMS commands.

Three parsers with a common signature parse(utterance) -> dict:

  - StubParser            : deterministic keyword/regex parser
  - SimulatedLLMParser    : richer pattern-based parser used as a
                            stand-in for an actual LLM (works offline
                            on Streamlit Cloud without API keys)
  - DirectParser          : maps utterance straight to theta, bypassing
                            structured intent and fuzzy mapping
                            (baseline for the ablation requested by
                            the reviewer)

If an OpenAI or Anthropic API key is configured in Streamlit secrets,
LLMParser uses the real API; otherwise it transparently falls back to
SimulatedLLMParser.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class Intent:
    guest_flag: int = 0
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    comfort_label: str = "neutral"
    comfort_priority: float = 0.5
    comfort_intensity: float = 1.0
    cost_label: str = "medium"
    budget: Optional[float] = None
    dr_label: str = "medium"
    dr_priority: float = 0.5
    medical_context: int = 0
    clarification_needed: int = 0
    parser_name: str = "stub"
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Stub parser - deterministic keyword matching
# ---------------------------------------------------------------------
GUEST_LEX   = {"guest", "guests", "company", "friends", "family", "party",
               "visitor", "visitors"}
WARM_LEX    = {"warm", "warmer", "cosy", "cozy", "heat", "heated", "hot"}
COOL_LEX    = {"cool", "cooler", "chilly", "cold", "ac"}
CHEAP_LEX   = {"cheap", "cheaper", "bill", "save", "money", "expensive",
               "afford"}
DR_LEX      = {"peak", "demand response", "dr event", "grid", "help during",
               "off-peak"}
MEDICAL_LEX = {"elderly", "grandma", "grandpa", "baby", "infant", "newborn",
               "sick", "ill", "asthma"}

INSISTENT = {"must", "essential", "critical", "absolutely", "no matter",
             "definitely"}
HEDGED    = {"try", "if possible", "kind of", "ish", "a bit", "slightly",
             "maybe"}


class StubParser:
    name = "stub"

    def parse(self, utterance: str) -> Intent:
        u = utterance.lower()
        words = set(re.findall(r"[a-z]+", u))

        intent = Intent(parser_name=self.name)

        if words & GUEST_LEX:
            intent.guest_flag = 1
        if words & WARM_LEX:
            intent.comfort_label = "warm"
            intent.comfort_priority = 0.7
        if words & COOL_LEX:
            intent.comfort_label = "cool"
            intent.comfort_priority = 0.5
        if words & CHEAP_LEX:
            intent.cost_label = "high"
        if any(p in u for p in DR_LEX):
            intent.dr_label = "high"
        if words & MEDICAL_LEX:
            intent.medical_context = 1
            intent.comfort_priority = max(intent.comfort_priority, 0.9)

        if any(w in u for w in INSISTENT):
            intent.comfort_intensity = 1.0
        elif any(w in u for w in HEDGED):
            intent.comfort_intensity = 0.4

        # Time window: "from 7 to 11", "after 7", "tonight"
        m = re.search(r"from\s+(\d{1,2})\s+to\s+(\d{1,2})", u)
        if m:
            intent.window_start = int(m.group(1))
            intent.window_end   = int(m.group(2))
        elif re.search(r"after\s+(\d{1,2})", u):
            mm = re.search(r"after\s+(\d{1,2})", u)
            intent.window_start = int(mm.group(1))
            intent.window_end   = 23

        # Clarification: vague time references or contradictions
        vague = any(t in u for t in ["tonight", "later", "soon", "evening",
                                     "around", "before bed"])
        conflict = (("max comfort" in u or "warm" in u) and
                    ("cheapest" in u or "lowest cost" in u))
        if (vague and intent.window_start is None) or conflict:
            intent.clarification_needed = 1
        return intent


# ---------------------------------------------------------------------
# Simulated LLM parser - the richer fallback used when no API key exists
# ---------------------------------------------------------------------
PARAPHRASTIC = {
    "warm": [r"crank\s+(?:the\s+)?heat", r"toast(y|i)", r"snug", r"don'?t\s+let.*freeze"],
    "cool": [r"cool\s+(?:it|down)", r"bring.*temperature.*down", r"air\s+con"],
    "save": [r"watch.*(?:the\s+)?bill", r"don'?t\s+spend", r"save.*money",
             r"keep.*low", r"on\s+a\s+budget", r"penny"],
    "dr":   [r"avoid.*peak", r"during.*expensive.*hours", r"off-?peak",
             r"help.*grid"],
}


class SimulatedLLMParser:
    """
    Drop-in stand-in for an LLM parser. Recognises paraphrastic and
    implicit phrasings that the stub parser misses. Always returns a
    schema-valid Intent so no fallback is triggered.
    """
    name = "llm-sim"

    def parse(self, utterance: str) -> Intent:
        u = utterance.lower()
        base = StubParser().parse(u)
        base.parser_name = self.name

        # Paraphrastic recovery
        for pat in PARAPHRASTIC["warm"]:
            if re.search(pat, u):
                base.comfort_label = "warm"
                base.comfort_priority = max(base.comfort_priority, 0.7)
        for pat in PARAPHRASTIC["cool"]:
            if re.search(pat, u):
                base.comfort_label = "cool"
        for pat in PARAPHRASTIC["save"]:
            if re.search(pat, u):
                base.cost_label = "high"
        for pat in PARAPHRASTIC["dr"]:
            if re.search(pat, u):
                base.dr_label = "high"

        # Implicit guest signals
        if any(p in u for p in ["dinner party", "people over", "mother-in-law",
                                "in-laws"]):
            base.guest_flag = 1

        # Italian / code-switched cues (Milan context)
        if any(t in u for t in ["bolletta", "caldo", "freddo", "ospiti",
                                "risparmiare"]):
            if "ospiti" in u:
                base.guest_flag = 1
            if "caldo" in u:
                base.comfort_label = "warm"
            if "freddo" in u or "fresco" in u:
                base.comfort_label = "cool"
            if "bolletta" in u or "risparmiare" in u:
                base.cost_label = "high"

        # Time window resolution
        if base.window_start is None:
            if "evening" in u or "dinner" in u or "tonight" in u:
                base.window_start, base.window_end = 19, 23
                base.clarification_needed = 0
            elif "morning" in u:
                base.window_start, base.window_end = 7, 11
            elif "after work" in u:
                base.window_start, base.window_end = 18, 22

        return base


# ---------------------------------------------------------------------
# Optional real LLM parser
# ---------------------------------------------------------------------
class LLMParser:
    """
    Schema-validated LLM parser. Tries to use OpenAI or Anthropic API
    if credentials exist; otherwise falls back to SimulatedLLMParser.
    """
    name = "llm"

    def __init__(self):
        self.simulated = SimulatedLLMParser()
        self._client = None
        self._backend = None
        try:
            import streamlit as st
            secrets = st.secrets
            if "OPENAI_API_KEY" in secrets:
                import openai
                self._client = openai.OpenAI(api_key=secrets["OPENAI_API_KEY"])
                self._backend = "openai"
            elif "ANTHROPIC_API_KEY" in secrets:
                import anthropic
                self._client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
                self._backend = "anthropic"
        except Exception:
            self._client = None

    def parse(self, utterance: str) -> Intent:
        if self._client is None:
            out = self.simulated.parse(utterance)
            out.parser_name = self.name + "-sim-fallback"
            return out

        prompt = self._build_prompt(utterance)
        try:
            if self._backend == "openai":
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=400,
                )
                text = resp.choices[0].message.content
            else:
                resp = self._client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text

            import json
            data = json.loads(text)
            intent = Intent(parser_name=self.name, **{
                k: v for k, v in data.items() if k in Intent.__annotations__
            })
            return intent
        except Exception:
            out = self.simulated.parse(utterance)
            out.parser_name = self.name
            out.fallback = True
            return out

    def _build_prompt(self, utterance: str) -> str:
        return f"""You are an intent parser for a home energy system.
Return a single JSON object with the fields:
  guest_flag (0/1), window_start (int 0-23 or null), window_end (int 0-23 or null),
  comfort_label (cool/neutral/warm/hot), comfort_priority (0-1),
  comfort_intensity (0-1, 1 = insistent, 0.3 = hedged),
  cost_label (low/medium/high), budget (number or null),
  dr_label (low/medium/high), dr_priority (0-1),
  medical_context (0/1), clarification_needed (0/1).

Utterance: "{utterance}"
JSON only, no commentary."""


# ---------------------------------------------------------------------
# Direct parser: utterance -> theta (no structured intent, no fuzzy)
# ---------------------------------------------------------------------
class DirectParser:
    """
    Baseline: skips structured intent + fuzzy mapping. Maps utterance
    straight to a theta dict. Used to ablate the modular pipeline.
    """
    name = "direct"

    def parse_to_theta(self, utterance: str) -> Dict[str, Any]:
        u = utterance.lower()
        T_tar = 21.5 if any(w in u for w in WARM_LEX) else \
                19.0 if any(w in u for w in COOL_LEX) else 20.5
        cost_w = 4.0 if any(w in u for w in CHEAP_LEX) else 1.5
        dr_w   = 4.0 if any(p in u for p in DR_LEX) else 1.5
        guest  = 1 if any(g in u for g in GUEST_LEX) else 0

        return {
            "T_target": T_tar,
            "T_min":    T_tar - (1.0 if guest else 2.0),
            "lambda_comf": 8.0 if guest else 3.0,
            "lambda_min":  20.0 if guest else 12.0,
            "lambda_cost": cost_w,
            "lambda_dr":   dr_w,
            "lambda_sw":   0.3,
            "guest_hard":  guest == 1,
            "_parser":     "direct",
        }


PARSERS = {
    "stub":   StubParser(),
    "llm":    LLMParser(),
    "direct": DirectParser(),
}
