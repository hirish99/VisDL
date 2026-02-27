"""REST API routes."""
import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..config import settings
from ..engine.executor import execute_graph
from ..engine.graph import Edge, Graph, NodeInstance
from ..engine.pipeline import execute_pipeline
from ..engine.validator import validate_graph
from ..models.schemas import (
    ExecuteRequest, ExecuteResponse, GraphSchema,
    SavedGraph, UploadResponse,
)
from ..engine.session import create_session, get_session, remove_session
from ..nodes.registry import NodeRegistry
from .websocket import manager

router = APIRouter(prefix="/api")
executor_pool = ThreadPoolExecutor(max_workers=4)

# In-memory stores
_results: dict[str, Any] = {}


def _schema_to_graph(schema: GraphSchema) -> Graph:
    nodes = {
        n.id: NodeInstance(
            id=n.id, node_type=n.node_type,
            params=n.params, disabled=n.disabled,
            position=n.position,
        )
        for n in schema.nodes
    }
    edges = [
        Edge(
            id=e.id, source_node=e.source_node, source_output=e.source_output,
            target_node=e.target_node, target_input=e.target_input, order=e.order,
        )
        for e in schema.edges
    ]
    return Graph(nodes=nodes, edges=edges)


@router.get("/nodes")
async def list_nodes():
    """Return all registered node definitions."""
    defs = NodeRegistry.all_definitions()
    result = {}
    for name, defn in defs.items():
        result[name] = {
            "node_type": defn.node_type,
            "display_name": defn.display_name,
            "category": defn.category,
            "description": defn.description,
            "inputs": {
                k: {
                    "dtype": v.dtype.value,
                    "default": v.default,
                    "required": v.required,
                    "min_val": v.min_val,
                    "max_val": v.max_val,
                    "choices": v.choices,
                    "is_handle": v.is_handle,
                }
                for k, v in defn.inputs.items()
            },
            "outputs": [
                {"dtype": o.dtype.value, "name": o.name}
                for o in defn.outputs
            ],
        }
    return result


@router.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """Execute the training pipeline: layer graph + config."""
    layer_graph = _schema_to_graph(request.graph)
    config = request.config.model_dump()

    session_id = request.session_id or str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    progress_cb = manager.make_progress_callback(session_id, loop)

    session = create_session(execution_id, session_id)
    checkpoint_path = settings.weights_dir / "checkpoints" / f"{execution_id}.pt"

    def run():
        return execute_pipeline(
            layer_graph, config,
            progress_callback=progress_cb,
            training_controller=session.controller,
            checkpoint_path=checkpoint_path,
        )

    try:
        await manager.send_to_session(session_id, {
            "type": "execution_start", "execution_id": execution_id,
        })

        results = await asyncio.get_event_loop().run_in_executor(executor_pool, run)

        serialized = _serialize_pipeline_results(results)
        _results[execution_id] = serialized

        await manager.send_to_session(session_id, {
            "type": "execution_complete", "execution_id": execution_id,
        })

        return ExecuteResponse(
            execution_id=execution_id, status="success", results=serialized,
        )
    except Exception as e:
        await manager.send_to_session(session_id, {
            "type": "execution_error", "error": str(e),
        })
        return ExecuteResponse(
            execution_id=execution_id, status="error", errors=[str(e)],
        )
    finally:
        remove_session(execution_id)


def _serialize_pipeline_results(results: dict[str, Any]) -> dict[str, Any]:
    """Convert pipeline results to JSON-serializable format."""
    serialized = {}
    for key, value in results.items():
        if isinstance(value, dict):
            clean = {}
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool, list, type(None))):
                    clean[k] = v
                elif isinstance(v, dict):
                    clean[k] = v
            serialized[key] = clean
        elif isinstance(value, (str, int, float, bool, list, type(None))):
            serialized[key] = value
        else:
            serialized[key] = str(type(value).__name__)
    return serialized


@router.post("/execute/{execution_id}/pause")
async def pause_training(execution_id: str):
    session = get_session(execution_id)
    if not session:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    session.controller.pause()
    await manager.send_to_session(session.session_id, {
        "type": "training_paused", "execution_id": execution_id,
    })
    return {"status": "paused"}


@router.post("/execute/{execution_id}/resume")
async def resume_training(execution_id: str):
    session = get_session(execution_id)
    if not session:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    session.controller.resume()
    await manager.send_to_session(session.session_id, {
        "type": "training_resumed", "execution_id": execution_id,
    })
    return {"status": "resumed"}


@router.post("/execute/{execution_id}/stop")
async def stop_training(execution_id: str):
    session = get_session(execution_id)
    if not session:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    session.controller.stop()
    await manager.send_to_session(session.session_id, {
        "type": "training_stopped", "execution_id": execution_id,
    })
    return {"status": "stopped"}


@router.get("/results/{execution_id}")
async def get_results(execution_id: str):
    if execution_id not in _results:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _results[execution_id]


@router.post("/upload/csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file, return file_id and column info."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_id = f"{uuid.uuid4()}_{file.filename}"
    dest = settings.upload_dir / file_id
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // (1024*1024)}MB). Max is {settings.max_upload_size_mb}MB.",
        )
    dest.write_bytes(content)

    df = pd.read_csv(dest)
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        columns=list(df.columns),
        rows=len(df),
    )


@router.post("/graphs")
async def save_graph(graph: SavedGraph):
    """Save a named graph configuration to disk."""
    if not graph.id:
        graph.id = str(uuid.uuid4())
    path = settings.graphs_dir / f"{graph.id}.json"
    path.write_text(graph.model_dump_json(indent=2))
    return {"id": graph.id}


@router.get("/graphs")
async def list_graphs():
    result = {}
    for path in settings.graphs_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            gid = data["id"]
            result[gid] = {"id": gid, "name": data.get("name", ""), "description": data.get("description", "")}
        except Exception:
            continue
    return result


@router.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    path = settings.graphs_dir / f"{graph_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    return json.loads(path.read_text())


@router.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    path = settings.graphs_dir / f"{graph_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    path.unlink()
    return {"status": "deleted"}
