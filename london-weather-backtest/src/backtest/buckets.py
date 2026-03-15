"""Temperature bucket logic mimicking Polymarket market structure."""

from __future__ import annotations

from config import (
    DEFAULT_BUCKET_SIZE,
    DEFAULT_NUM_BUCKETS,
    LONDON_MONTHLY_CLIMATOLOGY,
)


def get_climatology_center(month: int) -> float:
    """Return the climatological average max temp for a given month.

    Args:
        month: Month number (1-12).

    Returns:
        Approximate average max temperature in °C.
    """
    return LONDON_MONTHLY_CLIMATOLOGY[month]


def generate_buckets(
    center_temp: float,
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
) -> list[dict]:
    """Generate temperature buckets centered around the expected temperature.

    The first bucket is "X°C or below", the last is "Y°C or above",
    and middle buckets span bucket_size degrees each.

    Args:
        center_temp: Expected temperature to center buckets around.
        bucket_size: Width of each bucket in °C.
        num_buckets: Total number of buckets to generate.

    Returns:
        List of dicts with keys: 'min', 'max', 'label', 'index'.
        min/max are None for open-ended buckets.
    """
    # Calculate the range of middle buckets
    num_middle = num_buckets - 2  # minus the two open-ended buckets
    total_middle_range = num_middle * bucket_size

    # Center the middle buckets around center_temp (rounded to even)
    center_rounded = round(center_temp)
    low_bound = center_rounded - total_middle_range // 2
    # Ensure low_bound aligns to odd number (so buckets are like 9-10, 11-12, etc.)
    if low_bound % 2 == 0:
        low_bound -= 1

    buckets = []

    # First bucket: "X°C or below"
    first_max = low_bound - 1
    buckets.append({
        "min": None,
        "max": first_max,
        "label": f"{first_max}°C or below",
        "index": 0,
    })

    # Middle buckets
    for i in range(num_middle):
        b_min = low_bound + i * bucket_size
        b_max = b_min + bucket_size - 1
        buckets.append({
            "min": b_min,
            "max": b_max,
            "label": f"{b_min}-{b_max}°C",
            "index": i + 1,
        })

    # Last bucket: "Y°C or above"
    last_min = low_bound + num_middle * bucket_size
    buckets.append({
        "min": last_min,
        "max": None,
        "label": f"{last_min}°C or above",
        "index": num_buckets - 1,
    })

    return buckets


def assign_bucket(temp: float, buckets: list[dict]) -> dict | None:
    """Determine which bucket a temperature falls into.

    The temperature is first rounded to the nearest integer (matching
    Weather Underground / Polymarket resolution).

    Args:
        temp: Temperature in °C (will be rounded).
        buckets: List of bucket dicts from generate_buckets().

    Returns:
        The matching bucket dict, or None if no match found.
    """
    temp_rounded = round(temp)

    for bucket in buckets:
        b_min = bucket["min"]
        b_max = bucket["max"]

        if b_min is None and temp_rounded <= b_max:
            return bucket
        if b_max is None and temp_rounded >= b_min:
            return bucket
        if b_min is not None and b_max is not None and b_min <= temp_rounded <= b_max:
            return bucket

    return None


def generate_daily_buckets(month: int) -> list[dict]:
    """Generate buckets for a given month using climatology.

    Args:
        month: Month number (1-12).

    Returns:
        List of bucket dicts.
    """
    center = get_climatology_center(month)
    return generate_buckets(center)
