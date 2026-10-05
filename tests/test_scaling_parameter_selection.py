# Plateau and objective-screening regression tests.
"""Regression tests for objective preprocessing and scale selection."""
from argparse import Namespace
from pathlib import Path
import sys
import numpy as np
import pytest
from lambda_support_recovery.api import floor_objective_values
from lambda_support_recovery.penalty import pen_n
from lambda_support_recovery.selection import DimensionPath, build_dimension_path, select_plateau, select_persistent_plateau

def test_floor_applies_only_to_finite_raw_objectives():
    raw_objectives = np.array([1e-12, 1e-08, 2e-08, np.inf])
    (floored, mask) = floor_objective_values(raw_objectives, 1e-08)
    np.testing.assert_array_equal(mask, [True, True, False, False])
    np.testing.assert_array_equal(floored[:3], [0.0, 0.0, 2e-08])
    assert np.isinf(floored[3])
    np.testing.assert_array_equal(raw_objectives, [1e-12, 1e-08, 2e-08, np.inf])

def test_floor_makes_dimension_path_prefer_smallest_plateau_model():
    dimensions = np.array([5, 6, 7, 8])
    raw_objectives = np.array([0.0001, 8e-09, 1e-13, 2e-14])
    penalties = np.sqrt(dimensions)
    (floored, _) = floor_objective_values(raw_objectives, 1e-08)
    path = build_dimension_path(dimensions, floored, penalties)
    assert path.dimension_at(0.0) == 6

def test_plateau_path_allows_decreasing_support_count_penalties():
    dimensions = [1, 2, 3]
    objectives = [3.0, 2.0, 1.0]
    penalties = [1.0, 3.0, 2.0]
    with pytest.raises(ValueError, match='nondecreasing'):
        build_dimension_path(dimensions, objectives, penalties)
    path = build_dimension_path(dimensions, objectives, penalties, require_monotonic_penalty=False)
    for scale in (0.0, 1.0, 3.0, 10.0):
        expected = dimensions[int(np.argmin(np.asarray(objectives) + scale * np.asarray(penalties)))]
        assert path.dimension_at(scale) == expected

def test_recommendation_factor_controls_recommended_scale_and_dimension():
    result = select_plateau([1, 2, 3], [17.0, 1.0, 0.0], [1, 2, 3], recommendation_factor=0.1)
    assert result.minimal_scale == pytest.approx(4.0)
    assert result.recommendation_factor == pytest.approx(0.1)
    assert result.recommended_scale == pytest.approx(0.4)
    assert result.selected_dimension == 3

def test_recommendation_factor_defaults_to_two():
    result = select_plateau([1, 2, 3], [17.0, 1.0, 0.0], [1, 2, 3])
    assert result.recommendation_factor == pytest.approx(2.0)
    assert result.recommended_scale == pytest.approx(8.0)
    assert result.selected_dimension == 2

@pytest.mark.parametrize('recommendation_factor', [0.0, -1.0, np.inf, np.nan, 'bad'])
def test_recommendation_factor_must_be_positive(recommendation_factor):
    with pytest.raises(ValueError, match='recommendation_factor must be'):
        select_plateau([1, 2], [1.0, 0.0], [1.0, 2.0], recommendation_factor=recommendation_factor)

@pytest.mark.parametrize('objective_floor', [-1.0, np.inf, np.nan, 'bad'])
def test_floor_rejects_invalid_values(objective_floor):
    with pytest.raises(ValueError, match='finite, nonnegative'):
        floor_objective_values([1.0], objective_floor)

@pytest.mark.parametrize('scale', [1e-10, 1.0, 10000000000.0])
def test_plateau_prefers_absolute_width_despite_tiny_neighbor(scale):
    widths = [2.7, 0.72, 7.4, 0.013, 2.14]
    transitions = scale * np.exp(np.r_[0.0, np.cumsum(widths)])
    path = DimensionPath(breakpoints=(0.0, *transitions), dimensions=(7, 6, 5, 4, 3, 2, 1))
    result = select_persistent_plateau(path)
    assert result.succeeded
    assert result.dimension == 4
    assert result.log_width == pytest.approx(7.4)
    assert result.persistence_score == pytest.approx(7.4)
    assert result.runner_up_score == pytest.approx(2.7)
    assert result.score_margin == pytest.approx(4.7)
    assert result.center == pytest.approx(np.sqrt(transitions[2] * transitions[3]))

def test_plateau_accepts_single_bounded_interval_and_applies_recommendation():
    result = select_plateau([1, 2, 3], [17.0, 1.0, 0.0], [1, 2, 3])
    assert result.plateau_selection.dimension == 2
    assert result.plateau_selection.runner_up_score is None
    assert result.plateau_selection.score_margin is None
    assert result.minimal_scale == pytest.approx(4.0)
    assert result.recommended_scale == pytest.approx(8.0)
    assert result.selected_dimension == 2
    assert result.recommendation_within_plateau

def test_plateau_accepts_widest_log_width_below_one():
    path = DimensionPath(breakpoints=(0.0, 1.0, 1.2, 2.0), dimensions=(4, 3, 2, 1))
    result = select_persistent_plateau(path)
    assert result.succeeded
    assert result.dimension == 2
    assert result.log_width == pytest.approx(np.log(2.0 / 1.2))

def test_plateau_recommendation_can_leave_a_narrow_winning_interval():
    result = select_plateau([1, 2, 3], [3.0, 1.0, 0.0], [1, 2, 3])
    assert result.plateau_selection.dimension == 2
    assert result.minimal_scale == pytest.approx(np.sqrt(2.0))
    assert result.recommended_scale == pytest.approx(2.0 * np.sqrt(2.0))
    assert result.selected_dimension == 1
    assert not result.recommendation_within_plateau

def test_plateau_rejects_tied_largest_log_widths():
    path = DimensionPath(breakpoints=(0.0, 1.0, 2.0, 4.0), dimensions=(4, 3, 2, 1))
    result = select_persistent_plateau(path)
    assert not result.succeeded
    assert result.dimension is None
    assert result.score_margin == 0.0
    assert 'same largest absolute log-width' in result.failure_reason

@pytest.mark.parametrize('path', [DimensionPath((0.0,), (2,)), DimensionPath((0.0, 1.0), (2, 1))])
def test_plateau_requires_a_bounded_interval(path):
    result = select_persistent_plateau(path)
    assert not result.succeeded
    assert result.dimension is None
    assert 'At least one bounded plateau' in result.failure_reason
