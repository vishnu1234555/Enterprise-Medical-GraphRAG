def build_context(disease_name: str, evidence: list[dict], disease_description: str = None) -> str:
    """Format raw Neo4j records into a readable context block."""
    lines = [f"Disease Entity: {disease_name}"]
    if disease_description:
        lines.append(disease_description)
    lines.append("")
    lines.append("Supporting Evidence (Target | Score | Source):")
    lines.append("-" * 45)
    if not evidence:
        lines.append("  No clinical evidence or targets found for this entity in the current graph.")
    else:
        for rec in evidence:
            target = rec.get("target") or "Unknown"
            score = rec.get("score")
            src = rec.get("source_info")
            score_str = f"{float(score):.4f}" if score is not None else "N/A"
            src_str = str(src) if src not in (None, "") else "N/A"
            lines.append(f"  • {target:<30} | score: {score_str} | source: {src_str}")
    return "\n".join(lines)


def build_target_context(target_name: str, evidence: list[dict]) -> str:
    lines = [f"Target Entity: {target_name}", ""]
    lines.append("Associated Diseases (Disease | Score):")
    lines.append("-" * 45)
    if not evidence:
        lines.append("  No clinical evidence or diseases found for this target in the current graph.")
    else:
        for rec in evidence:
            disease = rec.get("disease") or "Unknown"
            score = rec.get("score")
            score_str = f"{float(score):.4f}" if score is not None else "N/A"
            lines.append(f"  • {disease:<30} | score: {score_str}")
    return "\n".join(lines)


def format_chat_history(chat_history: list | None) -> str:
    """Turn Streamlit message dicts into a single string for the LLM."""
    if not chat_history:
        return ""
    lines: list[str] = []
    for msg in chat_history:
        role = msg.get("role") or ""
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


def history_section(conversation_history: str) -> str:
    if not conversation_history.strip():
        return ""
    return (
        "## Prior conversation (dialogue memory — not verified medical evidence; "
        "use only to resolve pronouns and follow-ups)\n"
        f"{conversation_history}\n\n"
    )


def format_retrieval_diagnostics(stats: dict, vector_index: str) -> str:
    """Format retrieval diagnostics facts."""
    lines: list[str] = []
    try:
        total = stats.get('total', 0)
        named = stats.get('named', 0)
        sample = stats.get('sample', [])
        idx_visible = stats.get('idx_visible', False)
        
        lines.append(f"- `Disease` nodes (total): **{total}**")
        lines.append(f"- `Disease` with non-empty `name`: **{named}**")
        lines.append(
            f"- Vector index `{vector_index}` visible: **{'yes' if idx_visible else 'no'}**"
        )
        if sample:
            bits = [f"`{r.get('id')}` → {r.get('name')!r}" for r in sample]
            lines.append("- Sample rows: " + "; ".join(bits))
    except Exception as exc:
        lines.append(f"- Diagnostics formatting failed: `{exc}`")
    return "\n".join(lines)
