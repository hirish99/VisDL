#!/usr/bin/env python3
"""Generate training data for 2D exterior Laplace PDE via Method of Fundamental Solutions.

Each sample: random smooth closed curve, random MFS sources inside,
exact analytic solution evaluated on boundary sensors and exterior query points.

Two output formats:
  compact (default): boundary values + query coords + target (n_sensors + 3 columns)
                     Columns: s_0_g, s_1_g, ..., query_x, query_y, u
  full:              all sensor features + globals + query + target (n_sensors*8 + 7 columns)
                     Columns: s_0_x, s_0_y, s_0_g, ..., perimeter, area, ..., query_x, query_y, u

Usage:
    # Compact (DeepONet-ready):
    python scripts/generate_laplace_data.py --n_samples 1000 --n_sensors 64

    # Full (all features):
    python scripts/generate_laplace_data.py --n_samples 1000 --n_sensors 64 --format full
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Curve generation
# ---------------------------------------------------------------------------

def random_star_curve(rng: np.random.Generator, difficulty: str = "medium"):
    """Return Fourier coefficients for a random star-shaped curve.

    r(t) = r0 + sum_k  alpha_k cos(kt) + beta_k sin(kt)
    with 1/k^2 spectral decay to ensure smoothness.

    Returns (r0, alphas, betas) where alphas/betas are arrays of length K.
    """
    cfg = {"easy": (0.05, 3), "medium": (0.15, 5), "hard": (0.25, 8)}
    sigma, K = cfg.get(difficulty, cfg["medium"])

    r0 = rng.uniform(0.8, 1.5)
    alphas = np.array([rng.normal(0, sigma / (k * k)) for k in range(1, K + 1)])
    betas = np.array([rng.normal(0, sigma / (k * k)) for k in range(1, K + 1)])

    # Verify r(t) > 0 everywhere (reject self-intersecting curves)
    t_check = np.linspace(0, TWO_PI, 500, endpoint=False)
    r_vals = _eval_r(r0, alphas, betas, t_check)
    if np.any(r_vals <= 0.05):
        # Retry with smaller perturbation
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
    """Compute tangent, normal, and curvature at parameter values t.

    Returns (dx_dt, dy_dt, kappa, nx, ny) arrays.
    """
    r = _eval_r(r0, alphas, betas, t)
    # dr/dt
    dr = np.zeros_like(t)
    for k, (a, b) in enumerate(zip(alphas, betas), 1):
        dr += -a * k * np.sin(k * t) + b * k * np.cos(k * t)

    # d2r/dt2
    d2r = np.zeros_like(t)
    for k, (a, b) in enumerate(zip(alphas, betas), 1):
        d2r += -a * k * k * np.cos(k * t) - b * k * k * np.sin(k * t)

    # x(t) = r(t) cos(t), y(t) = r(t) sin(t)
    cost, sint = np.cos(t), np.sin(t)
    dx = dr * cost - r * sint
    dy = dr * sint + r * cost

    d2x = d2r * cost - 2 * dr * sint - r * cost
    d2y = d2r * sint + 2 * dr * cost - r * sint

    # Curvature: kappa = (dx * d2y - dy * d2x) / (dx^2 + dy^2)^(3/2)
    speed_sq = dx * dx + dy * dy
    speed = np.sqrt(speed_sq)
    kappa = (dx * d2y - dy * d2x) / (speed_sq * speed + 1e-30)

    # Outward unit normal (rotate tangent 90 degrees clockwise for outward-pointing)
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

    # Shoelace formula for area
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))

    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    return perimeter, area, centroid_x, centroid_y


# ---------------------------------------------------------------------------
# MFS: sources, Green's function, solution
# ---------------------------------------------------------------------------

def place_sources_inside(r0, alphas, betas, n_sources, rng):
    """Place MFS source points inside the curve (scaled boundary toward centroid)."""
    t_src = np.linspace(0, TWO_PI, n_sources, endpoint=False)
    t_src += rng.uniform(0, TWO_PI / n_sources)  # random offset
    x_bdy, y_bdy = evaluate_curve(r0, alphas, betas, t_src)

    _, _, cx, cy = curve_geometry(r0, alphas, betas)
    scale = rng.uniform(0.3, 0.7)
    x_src = cx + scale * (x_bdy - cx)
    y_src = cy + scale * (y_bdy - cy)
    return x_src, y_src


def greens_2d(px, py, sx, sy):
    """2D Laplace Green's function: G(p, s) = -1/(2*pi) * log|p - s|.

    px, py: evaluation points (N,)
    sx, sy: source points (M,)
    Returns: (N, M) matrix.
    """
    dx = px[:, None] - sx[None, :]
    dy = py[:, None] - sy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return -1.0 / TWO_PI * np.log(dist + 1e-30)


def evaluate_potential(qx, qy, sx, sy, strengths):
    """Evaluate u(query) = sum_j q_j * G(query, source_j)."""
    G = greens_2d(qx, qy, sx, sy)  # (n_query, n_sources)
    return G @ strengths


# ---------------------------------------------------------------------------
# Query point sampling
# ---------------------------------------------------------------------------

def sample_query_points(r0, alphas, betas, n_queries, rng):
    """Sample exterior query points biased toward near-field."""
    t_q = rng.uniform(0, TWO_PI, n_queries)
    x_bdy, y_bdy = evaluate_curve(r0, alphas, betas, t_q)
    _, _, _, nx, ny = curve_derivatives(r0, alphas, betas, t_q)

    # Exponential offset along outward normal
    offset = rng.exponential(scale=0.5, size=n_queries) + 0.05
    qx = x_bdy + offset * nx
    qy = y_bdy + offset * ny
    return qx, qy


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_sample(rng, n_sensors, n_queries, n_sources, difficulty, compact=True):
    """Generate one training sample.

    Returns list of rows (one per query point).
    compact=True:  boundary values g(sensor_i) + query coords + target
    compact=False: all 8 sensor features + global features + query + target
    """
    r0, alphas, betas = random_star_curve(rng, difficulty)

    # Boundary sensors (equally spaced in parameter)
    t_sensors = np.linspace(0, TWO_PI, n_sensors, endpoint=False)
    sx, sy = evaluate_curve(r0, alphas, betas, t_sensors)
    dx_dt, dy_dt, kappa, nx, ny = curve_derivatives(r0, alphas, betas, t_sensors)

    # MFS sources and random strengths
    src_x, src_y = place_sources_inside(r0, alphas, betas, n_sources, rng)
    strengths = rng.normal(0, 1.0, n_sources)

    # Boundary values at sensors
    g_vals = evaluate_potential(sx, sy, src_x, src_y, strengths)

    # Query points in exterior
    qx, qy = sample_query_points(r0, alphas, betas, n_queries, rng)
    u_vals = evaluate_potential(qx, qy, src_x, src_y, strengths)

    if compact:
        # Only boundary values (branch net input) + query coords (trunk net input) + target
        sensor_feats = list(g_vals)
    else:
        # Full sensor feature vectors: [x, y, g, dx, dy, kappa, nx, ny] per sensor
        sensor_feats = []
        for i in range(n_sensors):
            sensor_feats.extend([sx[i], sy[i], g_vals[i], dx_dt[i], dy_dt[i], kappa[i], nx[i], ny[i]])
        # Global features
        perimeter, area, centroid_x, centroid_y = curve_geometry(r0, alphas, betas)
        sensor_feats.extend([perimeter, area, centroid_x, centroid_y])

    rows = []
    for j in range(n_queries):
        row = sensor_feats + [qx[j], qy[j], u_vals[j]]
        rows.append(row)

    return rows


def build_header(n_sensors, compact=True):
    """Build CSV header row.

    compact=True:  s_0_g, s_1_g, ..., query_x, query_y, u
    compact=False: s_0_x, s_0_y, s_0_g, ..., perimeter, ..., query_x, query_y, u
    """
    cols = []
    if compact:
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


def main():
    parser = argparse.ArgumentParser(description="Generate Laplace PDE training data via MFS")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of boundary condition samples")
    parser.add_argument("--n_sensors", type=int, default=64, help="Boundary sensors per sample")
    parser.add_argument("--n_queries", type=int, default=100, help="Exterior query points per sample")
    parser.add_argument("--n_sources", type=int, default=20, help="MFS source points per sample")
    parser.add_argument("--output", type=str, default="data/uploads/laplace_train.csv", help="Output CSV path")
    parser.add_argument("--format", choices=["compact", "full"], default="compact",
                        help="compact: boundary values + query + target (default). full: all 8 sensor features + globals.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compact = args.format == "compact"
    rng = np.random.default_rng(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = build_header(args.n_sensors, compact=compact)
    total_rows = args.n_samples * args.n_queries

    print(f"Generating {args.n_samples} samples x {args.n_queries} queries = {total_rows} rows")
    print(f"  sensors={args.n_sensors}, sources={args.n_sources}, difficulty={args.difficulty}")
    fmt_desc = f"{args.n_sensors} boundary + 2 query + 1 target" if compact else f"{args.n_sensors}*8 sensor + 4 global + 2 query + 1 target"
    print(f"  format: {args.format}, columns: {len(header)} ({fmt_desc})")
    print(f"  output: {output_path}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(args.n_samples):
            rows = generate_sample(rng, args.n_sensors, args.n_queries, args.n_sources, args.difficulty, compact=compact)
            writer.writerows(rows)

            if (i + 1) % max(1, args.n_samples // 10) == 0:
                pct = 100 * (i + 1) / args.n_samples
                print(f"  [{pct:5.1f}%] {i + 1}/{args.n_samples} samples")

    print(f"Done. Wrote {total_rows} rows to {output_path}")
    if compact:
        print(f"\nVisDL config:")
        print(f"  input_columns: s_*_g,query_x,query_y")
        print(f"  target_columns: u")


if __name__ == "__main__":
    main()
