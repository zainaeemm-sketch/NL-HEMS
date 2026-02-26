def should_clarify(command: str, intent) -> tuple[bool, str]:
    text = command.lower()

    # Cases where ambiguity materially affects scheduling
    vague_time = any(k in text for k in ["later", "soon", "early"]) and "tonight" not in text
    strong_tradeoff_conflict = (
        ("warm" in text or "cozy" in text)
        and ("save" in text or "don't run up the bill" in text or "expensive" in text)
        and intent.guest_event
    )

    if strong_tradeoff_conflict:
        return True, "Do you want me to prioritize guest comfort over cost if prices spike tonight?"

    if vague_time:
        return True, "What time window should I optimize for?"

    return False, ""