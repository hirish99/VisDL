"""REST API routes."""
import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect

from ..config import settings
from ..engine.executor import execute_graph
from ..engine.graph import Edge, Graph, NodeInstance
from ..engine.validator import validate_graph
from ..models.schemas import (
    ExecuteRequest, ExecuteResponse, GraphSchema,
    SavedGraph, UploadResponse,
)
from ..nodes.registry import NodeRegistry
from .websocket import manager

router = APIRouter(prefix="/api")
executor_pool = ThreadPoolExecutor(max_workers=4)

# In-memory stores
_results: dict[str, Any] = {}
_saved_graphs: dict[str, SavedGraph] = {}


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
    """Validate and execute a graph."""
    graph = _schema_to_graph(request.graph)
    errors = validate_graph(graph)
    if errors:
        return ExecuteResponse(
            execution_id="", status="error", errors=errors,
        )

    session_id = request.session_id or str(uuid.uuid4())
    execution_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    progress_cb = manager.make_progress_callback(session_id, loop)

    def run():
        return execute_graph(graph, progress_callback=progress_cb)

    try:
        # Notify start
        await manager.send_to_session(session_id, {
            "type": "execution_start", "execution_id": execution_id,
        })

        results = await asyncio.get_event_loop().run_in_executor(executor_pool, run)

        # Serialize results (extract serializable data)
        serialized = _serialize_results(results)
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


def _serialize_results(results: dict[str, tuple]) -> dict[str, Any]:
    """Convert execution results to JSON-serializable format."""
    serialized = {}
    for node_id, outputs in results.items():
        node_outputs = []
        for output in outputs:
            if isinstance(output, dict):
                # Filter out non-serializable values
                clean = {}
                for k, v in output.items():
                    if isinstance(v, (str, int, float, bool, list, type(None))):
                        clean[k] = v
                    elif isinstance(v, dict):
                        clean[k] = v
                node_outputs.append(clean)
            elif isinstance(output, (str, int, float, bool, list, type(None))):
                node_outputs.append(output)
            else:
                node_outputs.append(str(type(output).__name__))
        serialized[node_id] = node_outputs
    return serialized


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
    """Save a named graph configuration."""
    if not graph.id:
        graph.id = str(uuid.uuid4())
    _saved_graphs[graph.id] = graph
    return {"id": graph.id}


@router.get("/graphs")
async def list_graphs():
    return {
        gid: {"id": g.id, "name": g.name, "description": g.description}
        for gid, g in _saved_graphs.items()
    }


@router.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    if graph_id not in _saved_graphs:
        raise HTTPException(status_code=404, detail="Graph not found")
    return _saved_graphs[graph_id]


@router.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    if graph_id not in _saved_graphs:
        raise HTTPException(status_code=404, detail="Graph not found")
    del _saved_graphs[graph_id]
    return {"status": "deleted"}
