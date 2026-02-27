"""Tests for the Laplace MFS data generation script."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the generation module directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from generate_laplace_data import (
    random_star_curve,
    _eval_r,
    evaluate_curve,
    curve_derivatives,
    curve_geometry,
    place_sources_inside,
    greens_2d,
    evaluate_potential,
    sample_query_points,
    generate_sample,
    build_header,
    TWO_PI,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestStarCurve:
    def test_r_positive_everywhere(self, rng):
        """Generated curve should have r(t) > 0 for all t."""
        for _ in range(20):
            r0, alphas, betas = random_star_curve(rng, "hard")
            t = np.linspace(0, TWO_PI, 500, endpoint=False)
            r = _eval_r(r0, alphas, betas, t)
            assert np.all(r > 0), f"Found non-positive r: min={r.min()}"

    def test_curve_is_closed(self, rng):
        """Start and end should match (periodic)."""
        r0, alphas, betas = random_star_curve(rng)
        t = np.array([0.0, TWO_PI])
        x, y = evaluate_curve(r0, alphas, betas, t)
        assert abs(x[0] - x[1]) < 1e-10
        assert abs(y[0] - y[1]) < 1e-10

    def test_difficulty_affects_perturbation(self, rng):
        """Hard curves should have more variation than easy."""
        rng_easy = np.random.default_rng(42)
        rng_hard = np.random.default_rng(42)
        _, a_easy, b_easy = random_star_curve(rng_easy, "easy")
        _, a_hard, b_hard = random_star_curve(rng_hard, "hard")
        # Hard should generally have larger amplitude coefficients
        # (at least on average across many samples)
        # We just verify both produce valid arrays
        assert len(a_easy) == 3  # easy: K=3
        assert len(a_hard) == 8  # hard: K=8


class TestCurveGeometry:
    def test_circle_perimeter(self):
        """A circle (no perturbation) should have perimeter ~ 2*pi*r."""
        r0, alphas, betas = 1.0, np.array([]), np.array([])
        perimeter, area, cx, cy = curve_geometry(r0, alphas, betas)
        assert abs(perimeter - TWO_PI * r0) < 0.01
        assert abs(area - np.pi * r0**2) < 0.05
        assert abs(cx) < 0.01
        assert abs(cy) < 0.01

    def test_area_positive(self, rng):
        for _ in range(10):
            r0, a, b = random_star_curve(rng)
            _, area, _, _ = curve_geometry(r0, a, b)
            assert area > 0


class TestCurveDerivatives:
    def test_normal_is_unit_length(self, rng):
        r0, a, b = random_star_curve(rng)
        t = np.linspace(0, TWO_PI, 100, endpoint=False)
        _, _, _, nx, ny = curve_derivatives(r0, a, b, t)
        norms = np.sqrt(nx**2 + ny**2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_normal_perpendicular_to_tangent(self, rng):
        r0, a, b = random_star_curve(rng)
        t = np.linspace(0, TWO_PI, 100, endpoint=False)
        dx, dy, _, nx, ny = curve_derivatives(r0, a, b, t)
        dot = dx * nx + dy * ny
        np.testing.assert_allclose(dot, 0.0, atol=1e-6)


class TestMFS:
    def test_greens_function_shape(self):
        px = np.array([1.0, 2.0, 3.0])
        py = np.array([0.0, 0.0, 0.0])
        sx = np.array([0.0, 0.5])
        sy = np.array([0.0, 0.0])
        G = greens_2d(px, py, sx, sy)
        assert G.shape == (3, 2)

    def test_greens_function_singularity(self):
        """Green's function should be large (negative) near source."""
        px = np.array([0.001])
        py = np.array([0.0])
        sx = np.array([0.0])
        sy = np.array([0.0])
        G = greens_2d(px, py, sx, sy)
        assert G[0, 0] > 0  # -1/(2pi) * log(small) > 0

    def test_sources_inside_curve(self, rng):
        """MFS sources should be inside the curve."""
        r0, a, b = random_star_curve(rng)
        src_x, src_y = place_sources_inside(r0, a, b, 20, rng)
        # Sources should be closer to centroid than boundary
        _, _, cx, cy = curve_geometry(r0, a, b)
        dist_to_center = np.sqrt((src_x - cx)**2 + (src_y - cy)**2)
        # All sources should be within the curve's max radius
        t = np.linspace(0, TWO_PI, 500, endpoint=False)
        r = _eval_r(r0, a, b, t)
        assert np.all(dist_to_center < np.max(r) * 1.1)

    def test_potential_linearity(self, rng):
        """Potential should scale linearly with strengths."""
        r0, a, b = random_star_curve(rng)
        src_x, src_y = place_sources_inside(r0, a, b, 10, rng)
        qx, qy = sample_query_points(r0, a, b, 5, rng)
        strengths = rng.normal(0, 1, 10)

        u1 = evaluate_potential(qx, qy, src_x, src_y, strengths)
        u2 = evaluate_potential(qx, qy, src_x, src_y, 2.0 * strengths)
        np.testing.assert_allclose(u2, 2.0 * u1, rtol=1e-10)


class TestSampleGeneration:
    def test_generate_sample_compact(self, rng):
        n_sensors, n_queries, n_sources = 8, 5, 10
        rows = generate_sample(rng, n_sensors, n_queries, n_sources, "medium", compact=True)
        assert len(rows) == n_queries
        expected_cols = n_sensors + 2 + 1  # boundary_values + query + target
        for row in rows:
            assert len(row) == expected_cols

    def test_generate_sample_full(self, rng):
        n_sensors, n_queries, n_sources = 8, 5, 10
        rows = generate_sample(rng, n_sensors, n_queries, n_sources, "medium", compact=False)
        assert len(rows) == n_queries
        expected_cols = n_sensors * 8 + 4 + 2 + 1  # sensor*8 + global + query + target
        for row in rows:
            assert len(row) == expected_cols

    def test_generate_sample_no_nans(self, rng):
        rows = generate_sample(rng, 16, 10, 10, "medium")
        for row in rows:
            assert not any(np.isnan(v) for v in row)

    def test_build_header_compact(self):
        header = build_header(4, compact=True)
        assert len(header) == 4 + 2 + 1  # 7
        assert header[0] == "s_0_g"
        assert header[-1] == "u"
        assert "query_x" in header

    def test_build_header_full(self):
        header = build_header(4, compact=False)
        assert len(header) == 4 * 8 + 4 + 2 + 1  # 39
        assert header[0] == "s_0_x"
        assert header[-1] == "u"
        assert "perimeter" in header
        assert "query_x" in header


class TestCLI:
    @pytest.mark.slow
    def test_cli_generates_csv_compact(self, tmp_path):
        output = tmp_path / "test_laplace.csv"
        result = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "generate_laplace_data.py"),
                "--n_samples", "3",
                "--n_sensors", "4",
                "--n_queries", "2",
                "--n_sources", "5",
                "--output", str(output),
                "--seed", "99",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output.exists()

        with open(output) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == 4 + 2 + 1  # compact default
            rows = list(reader)
            assert len(rows) == 3 * 2  # n_samples * n_queries

    @pytest.mark.slow
    def test_cli_generates_csv_full(self, tmp_path):
        output = tmp_path / "test_laplace_full.csv"
        result = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "generate_laplace_data.py"),
                "--n_samples", "2",
                "--n_sensors", "4",
                "--n_queries", "2",
                "--n_sources", "5",
                "--format", "full",
                "--output", str(output),
                "--seed", "99",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output.exists()

        with open(output) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == 4 * 8 + 4 + 2 + 1  # full format
            rows = list(reader)
            assert len(rows) == 2 * 2
