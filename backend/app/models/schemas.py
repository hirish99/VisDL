"""Pydantic schemas for API request/response models."""
from typing import Any
from pydantic import BaseModel


class NodeParamSchema(BaseModel):
    key: str
    value: Any


class EdgeSchema(BaseModel):
    id: str
    source_node: str
    source_output: int = 0
    target_node: str
    target_input: str
    order: int = 0


class NodeSchema(BaseModel):
    id: str
    node_type: str
    params: dict[str, Any] = {}
    disabled: bool = False
    position: dict[str, float] = {}


class GraphSchema(BaseModel):
    nodes: list[NodeSchema]
    edges: list[EdgeSchema]
    name: str = ""
    description: str = ""


class ExecuteRequest(BaseModel):
    graph: GraphSchema
    session_id: str | None = None


class ExecuteResponse(BaseModel):
    execution_id: str
    status: str
    results: dict[str, Any] = {}
    errors: list[str] = []


class NodeDefinitionResponse(BaseModel):
    node_type: str
    display_name: str
    category: str
    description: str
    inputs: dict[str, Any]
    outputs: list[dict[str, Any]]


class SavedGraph(BaseModel):
    id: str
    name: str
    description: str = ""
    graph: GraphSchema


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    columns: list[str]
    rows: int
