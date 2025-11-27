"""
Dynamic ZIP coverage sanity checks to avoid cherry-picked geolocation lists.

These tests sample random valid US ZIP codes from the pgeocode dataset on
every run (seeded for reproducibility) and assert the mapper returns
coordinates/state info. They are lightweight (no network calls to WQP/USGS).
"""

import random
import pytest
import pgeocode

from src.geolocation.zipcode_mapper import ZipCodeMapper


def _load_valid_zip_pool():
    nomi = pgeocode.Nominatim("US")
    df = nomi._data[["postal_code", "latitude", "longitude", "state_code"]].dropna()
    df = df[df["postal_code"].str.match(r"^\d{5}$")]
    return df


@pytest.fixture(scope="module")
def zip_pool():
    return _load_valid_zip_pool()


@pytest.fixture(scope="module")
def mapper():
    return ZipCodeMapper()


def test_random_zip_geolocation_is_valid(mapper, zip_pool):
    """Sample 100 random ZIPs and ensure we get usable coordinates."""
    rng = random.Random(20251127)
    samples = zip_pool.sample(n=100, random_state=rng.randint(0, 10_000))

    for _, row in samples.iterrows():
        z = row["postal_code"]
        assert mapper.is_valid_zipcode(z), f"{z} should be valid"
        coords = mapper.get_coordinates(z)
        assert coords is not None, f"{z} missing coordinates"
        lat, lon = coords
        assert -180 <= lon <= -50, f"{z} lon out of range {lon}"
        assert 17 <= lat <= 72, f"{z} lat out of range {lat}"


def test_random_zip_info_contains_state(mapper, zip_pool):
    """Randomly sample 50 ZIPs and check state_code is populated."""
    rng = random.Random(20251127)
    samples = zip_pool.sample(n=50, random_state=rng.randint(0, 10_000))

    for _, row in samples.iterrows():
        z = row["postal_code"]
        info = mapper.get_location_info(z)
        assert info is not None, f"{z} has no location info"
        assert info.get("state_code"), f"{z} missing state_code"
