# VisDL
<img width="1680" height="1050" alt="Screenshot 2026-02-26 at 1 11 00 AM" src="https://github.com/user-attachments/assets/9cd980fd-cf0c-45cb-9a5c-57bc1b8c30ac" />

Visual Deep Learning Research Tool — a ComfyUI-style web app for constructing, training, and ablating neural network architectures through a node graph UI.

## Quick Start

### Backend
```bash
cd backend
../.venv/bin/python run.py
```
Runs on http://localhost:8000

### Frontend
```bash
cd frontend
npm run dev -- --host 0.0.0.0
```
Runs on http://localhost:5173 (proxies API calls to backend)

### SSH Access
If accessing remotely, forward both ports:
```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@host
```

## Architecture

```
Browser (React Flow)  <->  FastAPI Backend (Python)
   |                          |
   +- Node Palette           +- Node Registry (auto-discovery)
   +- Canvas (drag/connect)  +- Graph Validator (DAG, types)
   +- Properties Panel       +- Execution Engine (topo sort)
   +- Training Dashboard     +- Training Loop (PyTorch, GPU)
   +- Ablation Panel         +- WebSocket (real-time progress)
```

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, Pydantic
- **Frontend**: React, TypeScript, React Flow, Zustand, Recharts, Vite
- **Communication**: REST for CRUD, WebSocket for real-time training telemetry

## Node Types (17)

| Category | Nodes |
|----------|-------|
| Data | CSV Loader, Data Splitter |
| Layers | Linear, ReLU, Sigmoid, Tanh, Dropout, BatchNorm1d |
| Model | Model Assembly |
| Loss | MSE Loss, Cross Entropy Loss, L1 Loss |
| Optimizer | SGD, Adam, AdamW |
| Training | Training Loop |
| Metrics | Metrics Collector |

## Adding New Nodes

Drop a decorated class in `backend/app/nodes/`:

```python
@NodeRegistry.register("MyNode")
class MyNode(BaseNode):
    CATEGORY = "MyCategory"
    DISPLAY_NAME = "My Node"

    @classmethod
    def INPUT_TYPES(cls):
        return {"x": InputSpec(dtype=DataType.TENSOR, required=True)}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.TENSOR, name="output")]

    def execute(self, **kwargs):
        return (kwargs["x"] * 2,)
```

Auto-discovered on startup. No registration boilerplate elsewhere.

## Key Design Decisions

- **Layer specs, not live modules**: Layer nodes emit spec dicts. ModelAssembly builds `nn.Sequential` with automatic shape inference.
- **GPU-aware**: Model placed on CUDA if available; training loop moves batches to model's device.
- **Ablation as first-class**: Every node has a disable toggle. Disabled layers become `nn.Identity`. Save/compare configs side-by-side.

## Sample Data

`sample_data.csv`, `sample_validation.csv`, `sample_test.csv` are included for testing. Function: `target = x1^2 + 2*x2 + noise`. Configure CSV Loader with `input_columns: x1,x2` and `target_columns: target`.

## Project Structure

```
backend/
  app/
    main.py              # FastAPI app, CORS, lifespan
    config.py            # Pydantic settings
    api/routes.py        # REST endpoints
    api/websocket.py     # WS connection manager
    engine/executor.py   # Topo sort + execute
    engine/validator.py  # DAG/type/input validation
    nodes/               # All node implementations (auto-discovered)
    models/schemas.py    # Pydantic request/response models
frontend/
  src/
    App.tsx              # Main layout
    store/               # Zustand (graphStore, executionStore)
    components/          # Canvas, NodePalette, NodeTypes, Panels
    hooks/               # useWebSocket, useNodeRegistry
    api/client.ts        # REST/WS client
```
