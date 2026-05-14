import json
from pathlib import Path

from core import (
    parse_command_stub,
    map_fuzzy_preferences,
    optimize_schedule,
    comfort_violation_minutes,
    dr_compliance_score,
)


def main():
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data" / "samples" / "demo_sample.json"

    if not sample_path.exists():
        sample_path.write_text(
            json.dumps(
                {
                    "sample_id": "NLHEMSDR_0001",
                    "command": "I have guests coming over tonight, keep the house warm but don't run up the bill.",
                }
            ),
            encoding="utf-8",
        )

    command = json.loads(sample_path.read_text(encoding="utf-8"))["command"]

    print("=== USER COMMAND ===")
    print(command)

    intent = parse_command_stub(command)
    print("\n=== PARSED INTENT ===")
    print(intent.model_dump_json(indent=2))

    mapped = map_fuzzy_preferences(intent)
    print("\n=== MAPPED PREFERENCES ===")
    print(json.dumps(mapped, indent=2))

    result = optimize_schedule(mapped)
    print("\n=== SCHEDULE RESULT ===")
    print(json.dumps(result, indent=2))

    if result["status"] == "ok":
        schedule = result["schedule"]
        print("\n=== METRICS ===")
        print(
            json.dumps(
                {
                    "comfort_violation_minutes": comfort_violation_minutes(schedule, mapped["min_temp_c"]),
                    "dr_compliance_score": dr_compliance_score(schedule),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()