"""
LandVerify — Multi-Agent Orchestrator + Dashboard Server
FastAPI app: serves index.html and streams agent results over WebSocket.
"""

import os
import json
import time
import random
import asyncio
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Config ────────────────────────────────────────────────────────────
API_BASE_URL         = os.environ.get("API_BASE_URL", "https://ailandcheck-production.up.railway.app")
API_VERIFY_URL       = f"{API_BASE_URL}/verify"
API_HEALTH_URL       = f"{API_BASE_URL}/health"
MAX_CONCURRENT       = int(os.environ.get("MAX_CONCURRENT_AGENTS", "4"))
REQUEST_TIMEOUT      = int(os.environ.get("REQUEST_TIMEOUT", "45"))
MAX_RETRIES          = int(os.environ.get("MAX_RETRIES", "2"))
CIRCUIT_BREAKER_MAX  = 3
PORT                 = int(os.environ.get("PORT", "8000"))
TEST_DOCS_DIR        = Path(os.environ.get("TEST_DOCS_DIR", "test_docs"))

# ── Data ──────────────────────────────────────────────────────────────
@dataclass
class AgentResult:
    agent_id: int
    document_path: str
    document_type: str
    expected_verdict: str
    actual_verdict: str
    trust_score: int
    squad_action: str
    llm_reasoning: str
    processing_time_ms: float
    correct: bool
    error: Optional[str] = None
    retries: int = 0

ACTIONS = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT", "ERROR": "ESCALATE"}

# ── Document Pool ─────────────────────────────────────────────────────
def get_document_pool() -> List[Dict]:
    base = TEST_DOCS_DIR
    candidates = [
        # Real documents (LOW risk)
        {"path": base / "real_benchmark.png",        "expected": "LOW",    "type": "Real C of O"},
        {"path": base / "real_benchmark - Copy.png", "expected": "LOW",    "type": "Real C of O"},
        {"path": base / "real_benchmark_2.png",      "expected": "LOW",    "type": "Real C of O"},
        {"path": base / "real_benchmark_2 - Copy.png","expected": "LOW",   "type": "Real C of O"},
        {"path": base / "real_fake.png",             "expected": "LOW",    "type": "Real C of O"},
        {"path": base / "real_fake - Copy.png",      "expected": "LOW",    "type": "Real C of O"},
        # Suspicious documents (MEDIUM risk)
        {"path": base / "closetoreal.png",           "expected": "MEDIUM", "type": "Suspicious Document"},
        {"path": base / "closetoreal - Copy.png",    "expected": "MEDIUM", "type": "Suspicious Document"},
        {"path": base / "closetoreal1.png",          "expected": "MEDIUM", "type": "Suspicious Document"},
        {"path": base / "closetoreal1 - Copy.png",   "expected": "MEDIUM", "type": "Suspicious Document"},
        {"path": base / "closetoreal2.png",          "expected": "MEDIUM", "type": "Suspicious Document"},
        {"path": base / "closetoreal2 - Copy.png",   "expected": "MEDIUM", "type": "Suspicious Document"},
        # Forged documents (HIGH risk)
        {"path": base / "fake.png",                  "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake - Copy.png",           "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake_another.png",          "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake_another - Copy.png",   "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake_another1.png",         "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake_another1 - Copy.png",  "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake_next.png",             "expected": "HIGH",   "type": "Forged Document"},
        {"path": base / "fake_next - Copy.png",      "expected": "HIGH",   "type": "Forged Document"},
    ]
    existing = [d for d in candidates if Path(d["path"]).exists()]
    if not existing:
        print("⚠️  No test_docs found — using mock pool")
        return _mock_pool()
    # Convert Path → str for JSON serialisation
    for d in existing:
        d["path"] = str(d["path"])
    return existing

def _mock_pool() -> List[Dict]:
    types = [("LOW", "Real C of O"), ("MEDIUM", "Suspicious Document"), ("HIGH", "Forged Document")]
    return [{"path": f"mock_{i}.png", "expected": t, "type": n} for i, (t, n) in enumerate(types * 4)]

# ── Agent Worker (runs in thread pool) ───────────────────────────────
def call_agent(assignment: Dict, agent_id: int) -> AgentResult:
    start = time.time()
    path            = assignment["path"]
    expected        = assignment["expected"]
    doc_type        = assignment.get("type", "Unknown")
    retries         = 0
    last_error      = None

    while retries <= MAX_RETRIES:
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    API_VERIFY_URL,
                    files={"file": (os.path.basename(path), f, "image/png")},
                    timeout=REQUEST_TIMEOUT,
                )
            if resp.status_code == 200:
                data = resp.json()
                verdict     = data.get("overall_risk", "UNKNOWN")
                score       = int(data.get("trust_score", 0))
                action      = data.get("squad_action", ACTIONS.get(verdict, "REVIEW"))
                reasoning   = data.get("recommendation", "Processed by verification engine")
                ms          = (time.time() - start) * 1000
                return AgentResult(
                    agent_id=agent_id, document_path=path, document_type=doc_type,
                    expected_verdict=expected, actual_verdict=verdict,
                    trust_score=score, squad_action=action, llm_reasoning=reasoning,
                    processing_time_ms=ms, correct=(verdict == expected),
                    error=None, retries=retries,
                )
            last_error = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_error = str(exc)

        retries += 1
        if retries <= MAX_RETRIES:
            time.sleep(2 ** retries)

    ms = (time.time() - start) * 1000
    return AgentResult(
        agent_id=agent_id, document_path=path, document_type=doc_type,
        expected_verdict=expected, actual_verdict="ERROR",
        trust_score=0, squad_action="ESCALATE", llm_reasoning="",
        processing_time_ms=ms, correct=False, error=last_error, retries=retries - 1,
    )

# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(title="LandVerify")

# Serve test_docs as static files (optional, for debugging)
if TEST_DOCS_DIR.exists():
    app.mount("/test_docs", StaticFiles(directory=str(TEST_DOCS_DIR)), name="test_docs")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/health")
async def health():
    # Check upstream API
    try:
        r = requests.get(API_HEALTH_URL, timeout=5)
        upstream = r.json() if r.status_code == 200 else {"status": f"HTTP {r.status_code}"}
    except Exception as e:
        upstream = {"status": "unreachable", "error": str(e)}
    docs = get_document_pool()
    return JSONResponse({"status": "ok", "docs_available": len(docs), "upstream_api": upstream})

@app.get("/docs-list")
async def docs_list():
    return JSONResponse({"documents": get_document_pool()})

# ── WebSocket — Real-time agent streaming ─────────────────────────────
@app.websocket("/ws/run")
async def run_agents(ws: WebSocket):
    await ws.accept()

    try:
        # Receive config from client
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        config = json.loads(raw)
        num_agents = max(1, min(int(config.get("num_agents", 7)), 20))
    except Exception:
        num_agents = 7

    await ws.send_json({"type": "config", "num_agents": num_agents, "max_concurrent": MAX_CONCURRENT})

    # Build assignment list
    pool = get_document_pool()
    while len(pool) < num_agents:
        pool.extend(pool)
    random.shuffle(pool)
    assignments = pool[:num_agents]

    await ws.send_json({"type": "start", "total": num_agents})

    loop = asyncio.get_event_loop()
    results = []
    consecutive_failures = 0
    circuit_open = False

    # Run batches with ThreadPoolExecutor (real I/O blocking calls)
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

    try:
        futures = {
            loop.run_in_executor(executor, call_agent, asgn, i + 1): (i + 1, asgn)
            for i, asgn in enumerate(assignments)
        }

        for coro in asyncio.as_completed(list(futures.keys())):
            if circuit_open:
                break
            try:
                result: AgentResult = await coro
                results.append(result)

                if result.error:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0

                await ws.send_json({"type": "agent_done", "result": asdict(result)})

                if consecutive_failures >= CIRCUIT_BREAKER_MAX:
                    circuit_open = True
                    await ws.send_json({
                        "type": "circuit_breaker",
                        "message": f"Circuit breaker triggered after {consecutive_failures} consecutive failures",
                    })

            except Exception as exc:
                consecutive_failures += 1
                await ws.send_json({"type": "agent_error", "error": str(exc)})

    finally:
        executor.shutdown(wait=False)

    # Summary
    valid   = [r for r in results if r.actual_verdict not in ("ERROR", "UNKNOWN")]
    correct = [r for r in valid if r.correct]
    accuracy = round(len(correct) / len(valid) * 100, 1) if valid else 0
    avg_trust = round(sum(r.trust_score for r in valid) / len(valid), 1) if valid else 0
    total_ms  = sum(r.processing_time_ms for r in results)

    await ws.send_json({
        "type": "complete",
        "summary": {
            "total_agents":    num_agents,
            "completed":       len(results),
            "correct":         len(correct),
            "accuracy":        accuracy,
            "avg_trust_score": avg_trust,
            "failures":        len([r for r in results if r.error]),
            "circuit_open":    circuit_open,
        },
    })

    try:
        await ws.close()
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
