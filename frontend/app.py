import streamlit as st
import httpx
import asyncio
import json
import subprocess
import atexit
import os

# Config
FASTAPI_CMD = ["uvicorn", "api.main:app", "--port", "8000", "--reload"]
FASTAPI_URL = "http://127.0.0.1:8000/query"

# Start Backend
if "backend_proc" not in st.session_state:
    st.session_state.backend_proc = subprocess.Popen(
        FASTAPI_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )

@atexit.register
def cleanup_backend():
    proc = st.session_state.get("backend_proc")
    if proc and proc.poll() is None:
        proc.terminate()

# Page Setup
st.set_page_config(page_title="Research Agent", layout="wide")

# Only target the markdown block inside each expander to be scrollable
st.markdown("""
<style>
/* Make the single markdown area inside each expander scrollable at a fixed height */
div[data-testid="stExpander"] .stMarkdown {
  max-height: 420px !important;
  overflow-y: auto !important;
  padding-right: 8px;
  display: block;
}

/* Header aesthetics + dark theme */
div[data-testid="stExpander"] div[role="button"] { font-weight:600; color:#58a6ff; }
body { background-color:#0d1117; color:#c9d1d9; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Research Agent — Streaming Workflow")

query = st.text_input("Enter your research query:", placeholder="e.g., vulnerabilities in LLMs")
run = st.button("Run Agent")

# main
async def run_agent(query_text: str):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", FASTAPI_URL, json={"query": query_text}) as resp:

            # Root collapsible sections (each has: status placeholder + ONE markdown placeholder)
            planner_exp = st.expander("🗺️ Planner", expanded=True)
            search_exp  = st.expander("🔎 Search & Retrieval", expanded=False)
            refl_exp    = st.expander("💭 Reflection", expanded=False)
            summ_exp    = st.expander("🧾 Summary", expanded=False)

            # Status (spinner text) placeholders
            planner_status = planner_exp.empty()
            search_status  = search_exp.empty()
            refl_status    = refl_exp.empty()
            summ_status    = summ_exp.empty()

            # Single markdown placeholders (scrollable via CSS above)
            planner_md = planner_exp.empty()
            search_md  = search_exp.empty()
            refl_md    = refl_exp.empty()
            summ_md    = summ_exp.empty()

            # Buffers for streaming text per section
            buffers = {"planner": "", "search": "", "reflection": "", "summary": ""}

            SECTION_MAP = {
                "planner": ("planner", planner_md, planner_status),
                "search_arxiv": ("search", search_md, search_status),
                "retrieval": ("search", search_md, search_status),
                "reflection": ("reflection", refl_md, refl_status),
                "summarize": ("summary", summ_md, summ_status),
            }

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.replace("data:", "").strip()
                if not data:
                    continue

                obj = json.loads(data)
                stage = obj.get("stage", "")
                msg = obj.get("message", "")

                # Ignore debug events
                if stage.startswith("debug_"):
                    continue

                # Start-of-stage (requires you to publish non-debug events in backend)
                if stage in ("planner", "search_arxiv", "reflection", "summarize"):
                    sec, _, stat = SECTION_MAP.get(stage, (None, None, None))
                    if not sec:
                        continue
                    stat.write(f"🌀 {msg or stage.title().replace('_',' ')}")
                    continue

                # Token streaming (Markdown)
                if stage.endswith("_token"):
                    sname = stage.replace("_token", "")
                    sec, ph, _ = SECTION_MAP.get(sname, (None, None, None))
                    if not sec:
                        continue
                    buffers[sec] += msg
                    ph.markdown(buffers[sec], unsafe_allow_html=True)
                    continue

                # End-of-stage
                if stage.endswith("_end"):
                    sname = stage.replace("_end", "")
                    sec, _, stat = SECTION_MAP.get(sname, (None, None, None))
                    if not sec:
                        continue
                    stat.empty()
                    continue

                # Final summary
                if "final_state" in obj:
                    summ_status.empty()
                    summ_md.markdown(obj["final_state"]["summary"], unsafe_allow_html=True)
                    st.success("✅ Agent completed successfully!")
                    return

# Run
if run and query:
    with st.spinner("Running Agent..."):
        asyncio.run(run_agent(query))
