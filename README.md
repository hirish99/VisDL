# VisDL

<img width="1629" height="965" alt="Screenshot 2026-02-26 at 1 19 23 AM" src="https://github.com/user-attachments/assets/f878093b-1868-4364-8447-770926801f45" />


Visual Deep Learning Research Tool — a ComfyUI-style web app for constructing, training, and ablating neural network architectures through a node graph UI.

## Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA (optional, but recommended for training)

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd VisDL
```

### 2. Create and activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note on PyTorch:** The `requirements.txt` lists `torch>=2.1.0`, which installs CPU-only by default. For GPU support, install PyTorch with CUDA separately first — see https://pytorch.org/get-started/locally/ for the correct command for your CUDA version. Example:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Running

You need **two terminals** (or use `&` / tmux / screen).

### Terminal 1 — Backend

```bash
cd VisDL
.venv/bin/python backend/run.py
```

The API server starts at **http://localhost:8000**.

### Terminal 2 — Frontend

```bash
cd VisDL/frontend
npm run dev
```

The UI starts at **http://localhost:5173**. Open this URL in your browser. The Vite dev server proxies `/api` and `/ws` requests to the backend automatically.

### Remote / SSH Access

If the server is on a remote machine, forward both ports:

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@host
```

Then open **http://localhost:5173** on your local machine.

## Architecture

```
Browser (React Flow)  <->  FastAPI Backend (Python)
   |                          |
   +- Node Palette           +- Node Registry (auto-discovery)
   +- Canvas (drag/connect)  +- Graph Validator (DAG, types)
   +- Properties Panel       +- Execution Engine (topo sort)
   +- Training Dashboard     +- Training Loop (PyTorch, GPU)
   +- System Status Bar      +- WebSocket (real-time progress)
```

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, Pydantic
- **Frontend**: React, TypeScript, React Flow, Zustand, Recharts, Vite
- **Communication**: REST for CRUD, WebSocket for real-time training telemetry and system monitoring

## Node Types (19)

| Category | Nodes |
|----------|-------|
| Data | CSV Loader, Data Splitter |
| Layers | Linear, ReLU, Sigmoid, Tanh, Dropout, BatchNorm1d |
| Model | Model Assembly, Model Export |
| Loss | MSE Loss, Cross Entropy Loss, L1 Loss |
| Optimizer | SGD, Adam, AdamW |
| Training | Training Loop |
| Metrics | Metrics Collector, Evaluator |

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
- **Pause/resume/stop training**: Thread-safe training controller with checkpoint save/restore. Pause mid-training, resume later, or stop early with partial results.
- **Large dataset support**: Chunked CSV reading for files >100MB, zero-copy train/val splitting via `torch.utils.data.Subset`, configurable upload size limit (default 500MB).
- **Live system monitoring**: CPU, RAM, GPU utilization, and VRAM usage streamed over WebSocket and displayed in the status bar.

## Sample Data

`sample_data.csv`, `sample_validation.csv`, `sample_test.csv` are included for testing. Function: `target = x1^2 + 2*x2 + noise`. Configure CSV Loader with `input_columns: x1,x2` and `target_columns: target`.

## Testing

```bash
# Run all tests (excluding slow/stress tests)
cd VisDL
.venv/bin/python -m pytest backend/tests/ -v

# Run stress tests for large datasets
.venv/bin/python -m pytest backend/tests/ -v -m slow
```

113 tests covering ablation, shape inference, model assembly, pipeline execution, graph validation, training control (pause/resume/stop, checkpoints), and large dataset handling.

## Project Structure

```
backend/
  run.py                          # Entry point (starts uvicorn)
  requirements.txt                # Python dependencies
  pytest.ini                      # Test configuration
  tests/                          # Test suite (113 tests)
  app/
    main.py                       # FastAPI app, CORS, lifespan, WS endpoints
    config.py                     # Pydantic settings
    api/routes.py                 # REST endpoints (incl. pause/resume/stop)
    api/websocket.py              # WS connection manager (training telemetry)
    api/system_monitor.py         # WS system stats (CPU/RAM/GPU)
    engine/executor.py            # Topo sort + execute
    engine/validator.py           # DAG/type/input validation
    engine/training_control.py    # Thread-safe pause/resume/stop controller
    engine/checkpoint.py          # Model checkpoint save/load
    engine/session.py             # Active execution session tracking
    nodes/                        # All node implementations (auto-discovered)
    models/schemas.py             # Pydantic request/response models
frontend/
  package.json                    # Node dependencies
  vite.config.ts                  # Dev server + proxy config
  src/
    App.tsx                       # Main layout
    store/                        # Zustand (graphStore, executionStore)
    components/                   # Canvas, NodePalette, NodeTypes, Panels, StatusBar, Toolbar
    hooks/                        # useWebSocket, useNodeRegistry, useSystemMonitor
    api/client.ts                 # REST/WS client
```
