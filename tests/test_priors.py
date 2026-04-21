"""Tests for metadata-based inference priors."""

from datetime import datetime

import torch

from wildlife_tools.inference.priors import (
    BaseLocationPrior,
    MovingLocationPrior,
    MultipleHomeLocationsPrior,
    TimeDecayPrior,
    grid_distance,
    parse_coords,
)


def test_parse_coords():
    """Grid coordinates should be parsed from the expected `x-y` string format."""
    assert parse_coords("10-20") == (10, 20)


def test_grid_distance():
    """Grid distance should match the floored Euclidean distance."""
    assert grid_distance("0-0", "3-4") == 5


def test_base_location_prior_prefers_nearby_identity():
    """A base location prior should upweight identities near the observed location."""
    prior = BaseLocationPrior([[("0-0", 3)], [("10-10", 2)]], alpha=1.0)
    appearance = torch.tensor([[0.5, 0.5]])
    output = prior(appearance, [{"grid_code": "0-0"}])

    assert output.shape == appearance.shape
    assert output[0, 0] > output[0, 1]


def test_multiple_home_locations_prior_uses_closest_location():
    """A multi-home prior should use the closest stored location for each identity."""
    prior = MultipleHomeLocationsPrior([[("0-0", 3), ("9-9", 1)], [("20-20", 2)]], alpha=1.0)
    appearance = torch.tensor([[0.5, 0.5]])
    output = prior(appearance, [{"grid_code": "8-8"}])

    assert output[0, 0] > output[0, 1]


def test_moving_location_prior_returns_normalized_probabilities():
    """The moving location prior should keep per-sample probabilities normalized."""
    prior = MovingLocationPrior([[("0-0", 3)], [("10-10", 2)]], alpha=1.0)
    appearance = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    output = prior(appearance, [{"grid_code": "0-0"}, {"grid_code": "0-0"}])

    assert torch.allclose(output.sum(dim=1), torch.ones(2))


def test_time_decay_prior_prefers_recent_year():
    """The time-decay prior should favor identities seen more recently."""
    prior = TimeDecayPrior([2023, 2019], alpha=1.0)
    appearance = torch.tensor([[0.5, 0.5]])
    output = prior(appearance, [{"timestamp": datetime(2023, 6, 1)}])

    assert output[0, 0] > output[0, 1]
