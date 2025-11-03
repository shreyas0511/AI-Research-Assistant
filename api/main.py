from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from agent import agent
from setup import *
from IPython.display import display, Markdown

import asyncio
import anyio
import contextlib

app = FastAPI(title="Research Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# request schema
class QueryRequest(BaseModel):
    query: str

class Event(BaseModel):
    stage: str
    message: str
    meta: dict | None = None

# root endpoint
@app.get("/")
def home():
    return {"message": "Research Agent API is running 🚀"}


@app.post("/query")
async def run_query(request: QueryRequest):
    """
    Handler function that handles the POST route to /query.
    Takes a user query and runs the agent.
    """
    # thread safe queue to handle tokens as they are streamed
    q: asyncio.Queue[Event] = asyncio.Queue()

    # function to put events into the queue
    async def publish(stage: str, message: str, meta: dict | None = None):
        await q.put(Event(stage=stage, message=message, meta=meta or {}))

    # initialize the graph state
    init_state = {
        "query": request.query,
        "original_plan": {},
        "plan": [],
        "results": {"arxiv": []},
        "reflection": None,
        "reflection_notes": "",
        "summary": "",
        "relevant_docs": [],
        "count": 0,
        "publish": publish,
    }

    async def event_stream():
        try:
            # run agent in background
            task = asyncio.create_task(
                agent.ainvoke(init_state, config={"recursion_limit": 200})
                if hasattr(agent, "ainvoke")
                else anyio.to_thread.run_sync(agent.invoke, init_state, {"recursion_limit": 100})
            )

            # stream queue items as they arrive
            while not task.done() or not q.empty():
                try:
                    event = await asyncio.wait_for(q.get(), timeout=0.2)
                    yield f"data: {json.dumps(event.dict())}\n\n"
                    q.task_done()
                except asyncio.TimeoutError:
                    if task.done() and q.empty():
                        break

            final_state = await task
            final_state.pop("publish", None)
            yield f"data: {json.dumps({'final_state': final_state})}\n\n"
            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ----- Run locally -----
if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
