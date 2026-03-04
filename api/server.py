import os
import sys
import json
import asyncio
import threading
import queue
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

# Load .env from project root (no-op in production where env vars are set directly)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to sys.path so `src.*` imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(
    title="Email Automation Dashboard",
    version="2.0",
    description="Real-time LangGraph Email Automation with live streaming UI",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Serve static files (the HTML dashboard)
STATIC_DIR = ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Health endpoint -- must respond INSTANTLY (no imports) ────────────────────
@app.get("/health")
async def health():
    """Lightweight health check for Railway/Render. No heavy imports here."""
    return {"status": "ok", "service": "Email Automation Dashboard"}


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Dashboard not found — static/index.html missing"}


# ── Node metadata sent to the browser ────────────────────────────────────────
NODE_META = {
    "load_inbox_emails":    {"icon": "📥", "color": "#6366f1", "label": "Load Inbox"},
    "is_email_inbox_empty": {"icon": "📭", "color": "#8b5cf6", "label": "Check Inbox"},
    "categorize_email":     {"icon": "🏷️",  "color": "#06b6d4", "label": "Categorize Email"},
    "construct_rag_queries":{"icon": "🔍", "color": "#10b981", "label": "Build RAG Queries"},
    "retrieve_from_rag":    {"icon": "📚", "color": "#f59e0b", "label": "Retrieve from RAG"},
    "email_writer":         {"icon": "✍️",  "color": "#f97316", "label": "Write Draft"},
    "email_proofreader":    {"icon": "🔎", "color": "#ec4899", "label": "Proofread Email"},
    "send_email":           {"icon": "📤", "color": "#22c55e", "label": "Send / Draft"},
    "skip_unrelated_email": {"icon": "⏭️", "color": "#94a3b8", "label": "Skip Unrelated"},
}


def _safe_serialize(obj):
    try:
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            return [_safe_serialize(i) for i in list(obj)]
        return str(obj) if not isinstance(obj, (int, float, bool, type(None))) else obj
    except Exception:
        return str(obj)


def _run_workflow(event_queue: queue.Queue):
    """Run the full LangGraph workflow in a background thread, pushing events to the queue."""
    try:
        # Lazy import — heavy ML deps only loaded when workflow is actually triggered
        from src.graph import Workflow

        config = {"recursion_limit": 100}
        workflow = Workflow()
        wf_app = workflow.app

        initial_state = {
            "emails": [],
            "current_email": {
                "id": "", "threadId": "", "messageId": "",
                "references": "", "sender": "", "subject": "", "body": ""
            },
            "email_category": "",
            "generated_email": "",
            "rag_queries": [],
            "retrieved_documents": "",
            "writer_messages": [],
            "sendable": False,
            "trials": 0,
        }

        event_queue.put({"type": "workflow_start", "text": "🚀 Workflow started!"})

        for output in wf_app.stream(initial_state, config):
            for node_name, state_value in output.items():
                meta = NODE_META.get(node_name, {"icon": "⚙️", "color": "#64748b", "label": node_name})
                event_queue.put({
                    "type": "node_complete",
                    "node": node_name,
                    "label": meta["label"],
                    "icon": meta["icon"],
                    "color": meta["color"],
                    "state": _safe_serialize(state_value),
                })

        event_queue.put({"type": "workflow_done", "text": "✅ Workflow completed successfully!"})

    except Exception as exc:
        event_queue.put({"type": "error", "text": f"❌ Error: {str(exc)}"})
    finally:
        event_queue.put(None)  # Sentinel


async def _stream_events(q: queue.Queue) -> AsyncGenerator[dict, None]:
    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await loop.run_in_executor(None, lambda: q.get(timeout=120))
        except Exception:
            break
        if item is None:
            break
        yield {"data": json.dumps(item)}
        await asyncio.sleep(0)


@app.get("/run")
async def run_workflow():
    """SSE endpoint — streams live workflow node events to the browser."""
    event_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_run_workflow, args=(event_queue,), daemon=True)
    thread.start()
    return EventSourceResponse(_stream_events(event_queue))
