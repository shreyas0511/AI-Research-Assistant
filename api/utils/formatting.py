"""
Formatting utilities for the Research Agent.
Converts raw JSON outputs from each node into clean, readable Markdown strings.
"""

import json
from typing import List, Dict, Any


# planner
def format_plan_for_display(plan_json: Dict[str, Any]) -> str:
    """Return Markdown-formatted plan."""
    if not plan_json or "plan" not in plan_json:
        return "⚠️ No plan generated."

    sections = []
    for i, step in enumerate(plan_json.get("plan", []), 1):
        tool = step.get("tool", "").replace("_", " ").title()
        purpose = step.get("purpose", "")
        rationale = step.get("rationale", "")
        query = step.get("query", {})

        terms = ", ".join(query.get("search_terms", []))
        focus = ", ".join(query.get("additional_focus", []))

        section = (
            f"#### Step {i}: {tool}\n"
            f"**Purpose:** {purpose}\n\n"
            f"**Search Terms:** {terms or '-'}\n\n"
            f"**Additional Focus:** {focus or '-'}\n\n"
            f"**Rationale:** {rationale or '-'}"
        )
        sections.append(section)

    reflection = (
        f"\n#### Reflection\n"
        f"**Purpose:** {plan_json["reflection"]["purpose"]}\n\n"
        f"**Analysis Foucs:**\n{'\n'.join([f"* {focus}" for focus in plan_json["reflection"]["analysis_focus"]])}\n\n"
        f"**Rationale:** {plan_json["reflection"]["rationale"]}\n\n"
    )

    sections.append(reflection)

    return "\n### 🗺️ Generated Plan\n\n" + "\n\n---\n\n".join(sections)


# search and retrieval
def format_search_queries(queries_dict: List[Dict[str, Any]], results: int) -> str:
    """Return Markdown showing search queries and retrieval summary."""
    text = "### 🔍 Search & Retrieval\n\n"

    if not queries_dict:
        return text + "⚠️ No queries generated."

    text += "#### Search Queries\n"
    for i, q in enumerate(queries_dict, 1):
        qtext = q.get("search_query", "").replace("\n", " ")
        maxres = q.get("max_results", "N/A")
        text += f"- **Query {i}:** `{qtext}` (max {maxres})\n"

    total = results if results else 0
    text += f"\n**Total papers retrieved:** {total}\n"

    return text


def format_retrieval_stats(total_docs: int, selected_docs: int, threshold: float) -> str:
    """Return Markdown summarizing retrieval stats."""
    return (
        "\n#### 📄 Relevance Filtering\n"
        f"- **Total retrieved:** {total_docs}\n"
        f"- **Selected (above threshold):** {selected_docs}\n"
        f"- **Threshold:** {threshold:.4f}\n"
    )


# reflection
def format_reflection(reflection_json: Dict[str, Any]) -> str:
    """Return Markdown summarizing reflection result."""
    if not reflection_json:
        return "⚠️ No reflection data."

    suff = reflection_json.get("sufficient", False)
    notes = reflection_json.get("notes", "")
    icon = "✅" if suff else "❌"
    verdict = "Sufficient papers found." if suff else "Needs more papers."

    return f"\n### 💭 Reflection Result\n{icon} **{verdict}**\n{notes or '-'}"


# summary
def format_summary(summary_text: str) -> str:
    """Return Markdown-formatted summary."""
    if not summary_text:
        return "⚠️ No summary generated."
    return f"\n### 🧾 Summary\n{summary_text}"


# generic
def pretty_json(obj: Any) -> str:
    """Utility: return a fenced JSON code block for quick debug display."""
    try:
        return f"```json\n{json.dumps(obj, indent=2)}\n```"
    except Exception:
        return str(obj)
