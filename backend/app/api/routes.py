"""REST API routes."""
import asyncio
import gc
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
    VramEstimateRequest, VramEstimateResponse,
)
from ..engine.session import create_session, get_session, remove_session
from ..nodes.registry import NodeRegistry
from .websocket import manager

router = APIRouter(prefix="/api")
executor_pool = ThreadPoolExecutor(max_workers=4)

# In-memory stores (capped at _MAX_RESULTS to prevent unbounded growth)
_MAX_RESULTS = 20
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
    """Execute the training pipeline: layer graph + config.

    Returns immediately with execution_id.  Training runs in background;
    progress, completion, and errors are delivered via WebSocket.
    """
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

    async def _run_training():
        try:
            await manager.send_to_session(session_id, {
                "type": "execution_start", "execution_id": execution_id,
            })

            results = await loop.run_in_executor(executor_pool, run)

            serialized = _serialize_pipeline_results(results)
            # Evict oldest entries if at capacity
            while len(_results) >= _MAX_RESULTS:
                _results.pop(next(iter(_results)))
            _results[execution_id] = serialized

            await manager.send_to_session(session_id, {
                "type": "execution_complete",
                "execution_id": execution_id,
                "results": serialized,
            })
        except Exception as e:
            await manager.send_to_session(session_id, {
                "type": "execution_error",
                "execution_id": execution_id,
                "error": str(e),
            })
        finally:
            remove_session(execution_id)
            # Free GPU memory even on exception
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    asyncio.create_task(_run_training())

    return ExecuteResponse(
        execution_id=execution_id, status="started", results={},
    )


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


@router.post("/estimate-vram", response_model=VramEstimateResponse)
async def estimate_vram(request: VramEstimateRequest):
    """Estimate GPU VRAM usage for the given architecture + config.

    Builds the model on CPU (no data needed) and calculates memory breakdown.
    """
    from ..engine.executor import topological_sort
    from ..engine.graph_module import trace_graph, infer_shapes_graph, GraphModule

    layer_graph = _schema_to_graph(request.graph)
    input_dim = request.input_dim
    batch_size = request.batch_size
    optimizer = request.optimizer

    # 1. Execute layer graph to get ArchRef
    order = topological_sort(layer_graph)
    layer_results: dict[str, tuple] = {}
    for node_id in order:
        node_inst = layer_graph.nodes[node_id]
        node_cls = NodeRegistry.get(node_inst.node_type)
        node = node_cls()
        node._node_id = node_id
        kwargs: dict = {}
        for edge in layer_graph.get_incoming_edges(node_id):
            src = layer_results.get(edge.source_node)
            if src is not None:
                kwargs[edge.target_input] = src[edge.source_output]
        for k, v in node_inst.params.items():
            if k not in kwargs:
                kwargs[k] = v
        if node_inst.disabled:
            layer_results[node_id] = node.on_disable(**kwargs)
        else:
            layer_results[node_id] = node.execute(**kwargs)

    # Find terminal node
    all_sources = {e.source_node for e in layer_graph.edges}
    terminal = None
    for nid in reversed(order):
        if nid not in all_sources:
            terminal = nid
            break
    if terminal is None:
        terminal = order[-1]
    arch_ref = layer_results[terminal][0]

    # 2. Build model on CPU
    all_nodes = trace_graph([arch_ref])
    infer_shapes_graph(all_nodes, input_dim)
    model = GraphModule(all_nodes, [arch_ref])

    # 3. Calculate memory
    MB = 1024 * 1024
    param_count = sum(p.numel() for p in model.parameters())
    params_mb = round(param_count * 4 / MB, 2)
    gradients_mb = params_mb
    optimizer_mult = 2.0 if optimizer in ("Adam", "AdamW") else 1.0
    optimizer_mb = round(params_mb * optimizer_mult, 2)

    # Activations: sum of per-node output features × batch_size × 4 bytes × 2 (fwd+bwd)
    act_elements = 0
    for node in all_nodes:
        for inp in node.inputs:
            pass  # just need output shapes
        # Estimate output size from shape inference: last out_features or num_features
        out_size = node.params.get("out_features") or node.params.get("num_features") or node.params.get("normalized_shape")
        if out_size is None and node.module_type == "DotProduct":
            out_size = 1
        if out_size is None and node.module_type == "Split":
            out_size = sum(node.params.get("split_sizes", []))
        if out_size is not None:
            act_elements += out_size
    activations_mb = round(act_elements * batch_size * 4 * 2 / MB, 2)

    # Batch data: input + target tensors
    last_linear = [n for n in all_nodes if n.module_type == "Linear"]
    output_dim = last_linear[-1].params.get("out_features", 1) if last_linear else 1
    batch_data_mb = round(batch_size * (input_dim + output_dim) * 4 / MB, 2)

    total_mb = round(params_mb + gradients_mb + optimizer_mb + activations_mb + batch_data_mb, 2)

    # 4. Query available GPU VRAM
    available_mb = None
    fits = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        available_mb = round(mem_info.total / MB, 0)
        fits = total_mb <= available_mb
    except Exception:
        pass

    return VramEstimateResponse(
        param_count=param_count,
        params_mb=params_mb,
        gradients_mb=gradients_mb,
        optimizer_mb=optimizer_mb,
        activations_mb=activations_mb,
        batch_data_mb=batch_data_mb,
        total_mb=total_mb,
        available_mb=available_mb,
        fits=fits,
    )


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
