#!/usr/bin/env python3
"""Generate training data for 2D PDE via MFS — GPU-accelerated.

Usage:
    # Compact format on GPU (default):
    python scripts/generate_laplace_data.py --n_samples 10000 --format compact --device cuda

    # Simple format on CPU:
    python scripts/generate_laplace_data.py --n_samples 1000 --format simple --device cpu

    # Helmholtz kernel:
    python scripts/generate_laplace_data.py --kernel helmholtz --wavenumber 5.0
"""
import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Legacy NumPy API  (backward-compatible — used by backend/tests/test_laplace_data.py)
# ---------------------------------------------------------------------------

TWO_PI = 2.0 * np.pi


def random_star_curve(rng: np.random.Generator, difficulty: str = "medium"):
    """Return Fourier coefficients for a random star-shaped curve."""
    cfg = {"easy": (0.05, 3), "medium": (0.15, 5), "hard": (0.25, 8)}
    sigma, K = cfg.get(difficulty, cfg["medium"])
    r0 = rng.uniform(0.8, 1.5)
    alphas = np.array([rng.normal(0, sigma / (k * k)) for k in range(1, K + 1)])
    betas = np.array([rng.normal(0, sigma / (k * k)) for k in range(1, K + 1)])
    t_check = np.linspace(0, TWO_PI, 500, endpoint=False)
    r_vals = _eval_r(r0, alphas, betas, t_check)
    if np.any(r_vals <= 0.05):
        return random_star_curve(rng, difficulty)
    return r0, alphas, betas


def _eval_r(r0, alphas, betas, t):
    """Evaluate radial function r(t)."""
    r = np.full_like(t, r0)
    for k, (a, b) in enumerate(zip(alphas, betas), 1):
        r += a * np.cos(k * t) + b * np.sin(k * t)
    return r


def evaluate_curve(r0, alphas, betas, t):
    """Return (x, y) positions on the curve at parameter values t."""
    r = _eval_r(r0, alphas, betas, t)
    return r * np.cos(t), r * np.sin(t)


def curve_derivatives(r0, alphas, betas, t):
    """Compute tangent, normal, and curvature at parameter values t."""
    r = _eval_r(r0, alphas, betas, t)
    dr = np.zeros_like(t)
    for k, (a, b) in enumerate(zip(alphas, betas), 1):
        dr += -a * k * np.sin(k * t) + b * k * np.cos(k * t)
    d2r = np.zeros_like(t)
    for k, (a, b) in enumerate(zip(alphas, betas), 1):
        d2r += -a * k * k * np.cos(k * t) - b * k * k * np.sin(k * t)
    cost, sint = np.cos(t), np.sin(t)
    dx = dr * cost - r * sint
    dy = dr * sint + r * cost
    d2x = d2r * cost - 2 * dr * sint - r * cost
    d2y = d2r * sint + 2 * dr * cost - r * sint
    speed_sq = dx * dx + dy * dy
    speed = np.sqrt(speed_sq)
    kappa = (dx * d2y - dy * d2x) / (speed_sq * speed + 1e-30)
    nx = dy / (speed + 1e-30)
    ny = -dx / (speed + 1e-30)
    return dx, dy, kappa, nx, ny


def curve_geometry(r0, alphas, betas, n_pts=1000):
    """Compute perimeter, area, and centroid of the curve."""
    t = np.linspace(0, TWO_PI, n_pts, endpoint=False)
    x, y = evaluate_curve(r0, alphas, betas, t)
    dx, dy, _, _, _ = curve_derivatives(r0, alphas, betas, t)
    dt = TWO_PI / n_pts
    speed = np.sqrt(dx * dx + dy * dy)
    perimeter = np.sum(speed) * dt
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    return perimeter, area, np.mean(x), np.mean(y)


def place_sources_inside(r0, alphas, betas, n_sources, rng):
    """Place MFS source points uniformly inside the curve via rejection sampling."""
    _, _, cx, cy = curve_geometry(r0, alphas, betas)
    t_check = np.linspace(0, TWO_PI, 500, endpoint=False)
    bx, by = evaluate_curve(r0, alphas, betas, t_check)
    x_min, x_max = bx.min(), bx.max()
    y_min, y_max = by.min(), by.max()
    x_src, y_src = [], []
    while len(x_src) < n_sources:
        batch = max(n_sources * 4, 100)
        cx_cand = rng.uniform(x_min, x_max, batch)
        cy_cand = rng.uniform(y_min, y_max, batch)
        dx = cx_cand - cx
        dy = cy_cand - cy
        theta = np.arctan2(dy, dx) % TWO_PI
        r_cand = np.sqrt(dx**2 + dy**2)
        r_boundary = _eval_r(r0, alphas, betas, theta)
        inside = r_cand < r_boundary * 0.9
        for i in np.where(inside)[0]:
            if len(x_src) >= n_sources:
                break
            x_src.append(cx_cand[i])
            y_src.append(cy_cand[i])
    return np.array(x_src), np.array(y_src)


def greens_2d(px, py, sx, sy):
    """2D Laplace Green's function: G(p, s) = -1/(2*pi) * log|p - s|."""
    dx = px[:, None] - sx[None, :]
    dy = py[:, None] - sy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return -1.0 / TWO_PI * np.log(dist + 1e-30)


def evaluate_potential(qx, qy, sx, sy, strengths):
    """Evaluate u(query) = sum_j q_j * G(query, source_j)."""
    G = greens_2d(qx, qy, sx, sy)
    return G @ strengths


def sample_query_points(r0, alphas, betas, n_queries, rng):
    """Sample exterior query points biased toward near-field."""
    t_q = rng.uniform(0, TWO_PI, n_queries)
    x_bdy, y_bdy = evaluate_curve(r0, alphas, betas, t_q)
    _, _, _, nx, ny = curve_derivatives(r0, alphas, betas, t_q)
    offset = rng.exponential(scale=0.5, size=n_queries) + 0.05
    return x_bdy + offset * nx, y_bdy + offset * ny


def generate_sample(rng, n_sensors, n_queries, n_sources, difficulty, fmt="simple"):
    """Generate one training sample (list of rows)."""
    r0, alphas, betas = random_star_curve(rng, difficulty)
    src_x, src_y = place_sources_inside(r0, alphas, betas, n_sources, rng)
    strengths = rng.normal(0, 1.0, n_sources)
    qx, qy = sample_query_points(r0, alphas, betas, n_queries, rng)
    u_vals = evaluate_potential(qx, qy, src_x, src_y, strengths)
    if fmt == "simple":
        return [[qx[j], qy[j], u_vals[j]] for j in range(n_queries)]
    t_sensors = np.linspace(0, TWO_PI, n_sensors, endpoint=False)
    sx, sy = evaluate_curve(r0, alphas, betas, t_sensors)
    dx_dt, dy_dt, kappa, nx, ny = curve_derivatives(r0, alphas, betas, t_sensors)
    g_vals = evaluate_potential(sx, sy, src_x, src_y, strengths)
    if fmt == "compact":
        sensor_feats = list(g_vals)
    else:
        sensor_feats = []
        for i in range(n_sensors):
            sensor_feats.extend([sx[i], sy[i], g_vals[i], dx_dt[i], dy_dt[i], kappa[i], nx[i], ny[i]])
        perimeter, area, centroid_x, centroid_y = curve_geometry(r0, alphas, betas)
        sensor_feats.extend([perimeter, area, centroid_x, centroid_y])
    rows = []
    for j in range(n_queries):
        rows.append(sensor_feats + [qx[j], qy[j], u_vals[j]])
    return rows


def build_header(n_sensors, fmt="simple"):
    """Build CSV header row."""
    if fmt == "simple":
        return ["x", "y", "u"]
    cols = []
    if fmt == "compact":
        for i in range(n_sensors):
            cols.append(f"s_{i}_g")
    else:
        feat_names = ["x", "y", "g", "dx", "dy", "k", "nx", "ny"]
        for i in range(n_sensors):
            for f in feat_names:
                cols.append(f"s_{i}_{f}")
        cols.extend(["perimeter", "area", "centroid_x", "centroid_y"])
    cols.extend(["query_x", "query_y"])
    cols.append("u")
    return cols


# ---------------------------------------------------------------------------
# GPU pipeline CLI
# ---------------------------------------------------------------------------

def main():
    import torch
    from pde_datagen.curves import FourierStarCurveGenerator
    from pde_datagen.kernels import LaplaceKernel2D, HelmholtzKernel2D
    from pde_datagen.sources import RejectionSourcePlacer
    from pde_datagen.queries import NormalOffsetQuerySampler
    from pde_datagen.features import SimpleAssembler, CompactAssembler, FullAssembler
    from pde_datagen.pipeline import BatchPipeline, PipelineConfig
    from pde_datagen.writer import write_csv

    parser = argparse.ArgumentParser(description="Generate 2D PDE training data (GPU-accelerated)")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_sensors", type=int, default=64)
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--min_sources", type=int, default=10,
                        help="Min MFS sources per curve")
    parser.add_argument("--max_sources", type=int, default=50,
                        help="Max MFS sources per curve")
    parser.add_argument("--output", type=str, default="data/uploads/laplace_train.csv")
    parser.add_argument("--format", choices=["simple", "compact", "full"], default="simple")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Curves processed simultaneously on GPU")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--kernel", choices=["laplace", "helmholtz"], default="laplace")
    parser.add_argument("--wavenumber", type=float, default=1.0,
                        help="Wavenumber k for Helmholtz kernel")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Kernel
    if args.kernel == "helmholtz":
        kernel = HelmholtzKernel2D(wavenumber=args.wavenumber)
    else:
        kernel = LaplaceKernel2D()

    # Feature assembler
    asm_map = {"simple": SimpleAssembler, "compact": CompactAssembler, "full": FullAssembler}
    feature_asm = asm_map[args.format]()

    config = PipelineConfig(
        n_samples=args.n_samples,
        n_sensors=args.n_sensors,
        n_queries=args.n_queries,
        min_sources=args.min_sources,
        max_sources=args.max_sources,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    total_rows = config.n_samples * config.n_queries
    header = feature_asm.header(config.n_sensors)

    print(f"Generating {config.n_samples} samples x {config.n_queries} queries = {total_rows} rows")
    print(f"  sensors={config.n_sensors}, sources={config.min_sources}-{config.max_sources}, difficulty={args.difficulty}")
    print(f"  format: {args.format}, columns: {len(header)}")
    print(f"  device: {device}, batch_size: {config.batch_size}, kernel: {args.kernel}")

    pipeline = BatchPipeline(
        curve_gen=FourierStarCurveGenerator(args.difficulty),
        source_placer=RejectionSourcePlacer(),
        query_sampler=NormalOffsetQuerySampler(),
        kernel=kernel,
        feature_asm=feature_asm,
        config=config,
    )

    def _progress(done, total):
        pct = 100 * done / total
        print(f"  [{pct:5.1f}%] {done}/{total} samples", flush=True)

    data = pipeline.generate(progress_cb=_progress)

    output_path = Path(args.output)
    write_csv(data, header, output_path)

    print(f"Done. Wrote {len(data)} rows to {output_path}")
    if args.format == "simple":
        print(f"\nVisDL config:\n  input_columns: x,y\n  target_columns: u")
    elif args.format == "compact":
        print(f"\nVisDL config:\n  input_columns: s_*_g,query_x,query_y\n  target_columns: u")


if __name__ == "__main__":
    main()
