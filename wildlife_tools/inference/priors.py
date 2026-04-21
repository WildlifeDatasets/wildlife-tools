from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any

import torch


def parse_coords(coord_string: str) -> tuple[int, int]:
    """Parse a grid code formatted as ``"x-y"`` into integer coordinates."""
    coord = coord_string.split("-")
    if len(coord) != 2:
        raise ValueError(f"Invalid coordinate string: {coord_string!r}")
    return int(coord[0]), int(coord[1])


def grid_distance(a: str, b: str) -> int:
    """Compute Euclidean distance between two grid codes and round it down."""
    a_x, a_y = parse_coords(a)
    b_x, b_y = parse_coords(b)
    return math.floor(math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2))


def date_distance(timestamp: datetime, year: int) -> int:
    """Return the absolute difference between a timestamp year and a reference year."""
    return abs(timestamp.year - year)


class BaseLocationPrior:
    """
    Reweight appearance probabilities by distance to a single home location per identity.

    This prior is adapted from the foreground-background modeling method introduced in
    "Animal Identification with Independent Foreground and Background Modeling"
    by Picek, Neumann, and Matas.

    Args:
        identity_to_base_location_map: Per-identity list of `(grid_code, count)` tuples ordered
            so that the first entry represents the primary location.
        alpha: Exponential decay strength for distance-based reweighting.
        threshold: Distance below which no penalty is applied.
    """

    def __init__(self, identity_to_base_location_map: list[list[tuple[str, int]]], alpha: float, threshold: int = 0):
        self.identity_to_base_location_map = identity_to_base_location_map
        self.alpha = alpha
        self.threshold = threshold

    def __call__(self, appearance_prob: torch.Tensor, metadata: list[dict[str, Any]]) -> torch.Tensor:
        """
        Apply the prior to appearance probabilities.

        Args:
            appearance_prob: Tensor of shape `(n_samples, n_classes)` with appearance-based probabilities.
            metadata: Per-sample metadata dictionaries containing `grid_code`.

        Returns:
            Tensor of reweighted probabilities with the same shape as `appearance_prob`.
        """
        base_location_dist = torch.zeros_like(appearance_prob)

        for sample_ix, sample_meta_data in enumerate(metadata):
            location = sample_meta_data["grid_code"]
            for identity_id, base_location in enumerate(self.identity_to_base_location_map):
                base_location_dist[sample_ix, identity_id] = max(
                    grid_distance(location, base_location[0][0]) - self.threshold, 0
                )

        base_location_prob = self.alpha * torch.exp(-base_location_dist * self.alpha)
        return appearance_prob * base_location_prob.to(appearance_prob.device)


class MultipleHomeLocationsPrior:
    """
    Reweight appearance probabilities by the nearest of multiple home locations per identity.

    This prior is adapted from the foreground-background modeling method introduced in
    "Animal Identification with Independent Foreground and Background Modeling"
    by Picek, Neumann, and Matas.

    Args:
        identity_to_base_location_map: Per-identity list of `(grid_code, count)` tuples.
        alpha: Exponential decay strength for distance-based reweighting.
        threshold: Distance below which no penalty is applied.
    """

    def __init__(self, identity_to_base_location_map: list[list[tuple[str, int]]], alpha: float, threshold: int = 0):
        self.identity_to_base_location_map = identity_to_base_location_map
        self.alpha = alpha
        self.threshold = threshold

    def __call__(self, appearance_prob: torch.Tensor, metadata: list[dict[str, Any]]) -> torch.Tensor:
        """
        Apply the prior to appearance probabilities.

        Args:
            appearance_prob: Tensor of shape `(n_samples, n_classes)` with appearance-based probabilities.
            metadata: Per-sample metadata dictionaries containing `grid_code`.

        Returns:
            Tensor of reweighted probabilities with the same shape as `appearance_prob`.
        """
        base_location_dist = torch.zeros_like(appearance_prob)

        for sample_ix, sample_meta_data in enumerate(metadata):
            location = sample_meta_data["grid_code"]
            for identity_id, base_locations in enumerate(self.identity_to_base_location_map):
                min_distance = 1000
                for base_location in base_locations:
                    min_distance = min(max(grid_distance(location, base_location[0]) - self.threshold, 0), min_distance)
                base_location_dist[sample_ix, identity_id] = min_distance

        base_location_prob = self.alpha * torch.exp(-base_location_dist * self.alpha)
        return appearance_prob * base_location_prob.to(appearance_prob.device)


class MovingLocationPrior:
    """
    Reweight appearance probabilities using a single location per identity that can move over time.

    This prior is adapted from the foreground-background modeling method introduced in
    "Animal Identification with Independent Foreground and Background Modeling"
    by Picek, Neumann, and Matas.

    Args:
        identity_to_base_location_map: Per-identity list of `(grid_code, count)` tuples ordered
            so that the first entry represents the starting location.
        alpha: Exponential decay strength for distance-based reweighting.
        location_update_prob_threshold: Minimum posterior probability required to update an
            identity's current location.
    """

    def __init__(
        self,
        identity_to_base_location_map: list[list[tuple[str, int]]],
        alpha: float,
        location_update_prob_threshold: float = 0.5,
    ):
        self.identity_to_base_location_map = identity_to_base_location_map
        self.alpha = alpha
        self.location_update_prob_threshold = location_update_prob_threshold

    def __call__(self, appearance_prob: torch.Tensor, metadata: list[dict[str, Any]]) -> torch.Tensor:
        """
        Apply the prior sequentially and update locations from confident predictions.

        Args:
            appearance_prob: Tensor of shape `(n_samples, n_classes)` with appearance-based probabilities.
            metadata: Per-sample metadata dictionaries containing `grid_code`.

        Returns:
            Tensor of reweighted probabilities with the same shape as `appearance_prob`.
        """
        identity_to_base_location_map = [loc[0][0] for loc in copy.deepcopy(self.identity_to_base_location_map)]
        num_identities = appearance_prob.size(1)
        prob = torch.zeros_like(appearance_prob)

        for sample_ix, sample_meta_data in enumerate(metadata):
            location = sample_meta_data["grid_code"]
            base_location_dist = torch.zeros(num_identities, dtype=appearance_prob.dtype)

            for identity_id, base_location in enumerate(identity_to_base_location_map):
                base_location_dist[identity_id] = max(grid_distance(location, base_location), 0)

            base_location_prob = torch.exp(-base_location_dist * self.alpha)
            base_location_prob = base_location_prob / base_location_prob.sum()

            prob[sample_ix, :] = appearance_prob[sample_ix, :] * base_location_prob.to(appearance_prob.device)
            prob[sample_ix, :] = prob[sample_ix, :] / prob[sample_ix, :].sum()

            pred_label = torch.argmax(prob[sample_ix, :]).item()
            pred_label_prob = prob[sample_ix, pred_label].item()
            if pred_label_prob > self.location_update_prob_threshold:
                identity_to_base_location_map[pred_label] = location

        return prob


class MultipleMovingLocationsPrior:
    """
    Reweight appearance probabilities using multiple candidate locations per identity.

    This prior is adapted from the foreground-background modeling method introduced in
    "Animal Identification with Independent Foreground and Background Modeling"
    by Picek, Neumann, and Matas.

    Args:
        identity_to_base_location_map: Per-identity list of `(grid_code, count)` tuples.
        alpha: Exponential decay strength for distance-based reweighting.
        observation_count_update_threshold: Number of repeated confident observations needed
            before adding a new location.
    """

    def __init__(
        self,
        identity_to_base_location_map: list[list[tuple[str, int]]],
        alpha: float,
        observation_count_update_threshold: int = 2,
    ):
        self.identity_to_base_location_map = identity_to_base_location_map
        self.alpha = alpha
        self.observation_count_update_threshold = observation_count_update_threshold

    def __call__(self, appearance_prob: torch.Tensor, metadata: list[dict[str, Any]]) -> torch.Tensor:
        """
        Apply the prior sequentially and add repeated new locations to the identity map.

        Args:
            appearance_prob: Tensor of shape `(n_samples, n_classes)` with appearance-based probabilities.
            metadata: Per-sample metadata dictionaries containing `grid_code`.

        Returns:
            Tensor of reweighted probabilities with the same shape as `appearance_prob`.
        """
        identity_to_base_location_map = copy.deepcopy(self.identity_to_base_location_map)
        num_identities = appearance_prob.size(1)
        previous_positions = {k: [] for k in range(num_identities)}
        prob = torch.zeros_like(appearance_prob)

        for sample_ix, sample_meta_data in enumerate(metadata):
            location = sample_meta_data["grid_code"]
            base_location_dist = torch.zeros(num_identities, dtype=appearance_prob.dtype)

            for identity_id, base_locations in enumerate(identity_to_base_location_map):
                min_distance = 1000
                for base_location in base_locations:
                    min_distance = min(max(grid_distance(location, base_location[0]), 0), min_distance)
                base_location_dist[identity_id] = min_distance

            base_location_prob = self.alpha * torch.exp(-base_location_dist * self.alpha)
            prob[sample_ix, :] = appearance_prob[sample_ix, :] * base_location_prob.to(appearance_prob.device)

            pred_label = torch.argmax(prob[sample_ix, :]).item()
            previous_positions[pred_label].append(location)

            if len(previous_positions[pred_label]) >= self.observation_count_update_threshold:
                locations = previous_positions[pred_label][: self.observation_count_update_threshold]
                if all(locations[0] == loc for loc in locations):
                    identity_to_base_location_map[pred_label].append((location, 1))

        return prob


class TimeDecayPrior:
    """
    Reweight appearance probabilities using the last known observation year per identity.

    This prior is adapted from the foreground-background modeling method introduced in
    "Animal Identification with Independent Foreground and Background Modeling"
    by Picek, Neumann, and Matas.

    Args:
        identity_to_last_year_map: Per-identity most recent observation year.
        alpha: Exponential decay strength for year-based reweighting.
        threshold: Year difference below which no penalty is applied.
        year_update_threshold: Number of repeated confident observations needed before updating
            an identity's last seen year.
    """

    def __init__(
        self,
        identity_to_last_year_map: list[int],
        alpha: float,
        threshold: int = 0,
        year_update_threshold: int = 1,
    ):
        self.identity_to_last_year_map = identity_to_last_year_map
        self.alpha = alpha
        self.threshold = threshold
        self.year_update_threshold = year_update_threshold

    def __call__(self, appearance_prob: torch.Tensor, metadata: list[dict[str, Any]]) -> torch.Tensor:
        """
        Apply the prior sequentially and update years from repeated confident predictions.

        Args:
            appearance_prob: Tensor of shape `(n_samples, n_classes)` with appearance-based probabilities.
            metadata: Per-sample metadata dictionaries containing `timestamp`.

        Returns:
            Tensor of reweighted probabilities with the same shape as `appearance_prob`.
        """
        identity_to_last_year_map = copy.deepcopy(self.identity_to_last_year_map)
        num_identities = appearance_prob.size(1)
        previous_years = {k: [] for k in range(num_identities)}
        prob = torch.zeros_like(appearance_prob)

        for sample_ix, sample_meta_data in enumerate(metadata):
            timestamp = sample_meta_data["timestamp"]
            last_year_dist = torch.zeros(num_identities, dtype=appearance_prob.dtype)

            for identity_id, last_year in enumerate(identity_to_last_year_map):
                last_year_dist[identity_id] = max(date_distance(timestamp, last_year) - self.threshold, 0)

            base_location_prob = self.alpha * torch.exp(-last_year_dist * self.alpha)
            prob[sample_ix, :] = appearance_prob[sample_ix, :] * base_location_prob.to(appearance_prob.device)

            pred_label = torch.argmax(prob[sample_ix, :]).item()
            previous_years[pred_label].append(timestamp.year)

            if len(previous_years[pred_label]) > self.year_update_threshold:
                years = previous_years[pred_label][: self.year_update_threshold + 1]
                if all(years[0] == year for year in years):
                    identity_to_last_year_map[pred_label] = timestamp.year

        return prob
