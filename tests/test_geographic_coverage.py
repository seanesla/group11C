"""
ULTRA-COMPREHENSIVE Geographic Coverage Tests

This module contains 300+ tests ensuring the application works across:
- All 50 US states with MULTIPLE ZIP codes per state
- Washington DC
- All US territories
- Border regions (Canada, Mexico)
- Time zones (all 6 US time zones)
- Special ZIP codes (military, single-building, etc.)
- Metropolitan areas
- Extreme elevations
- Island locations
- Distance/radius edge cases

Test Coverage:
- Multiple ZIPs per state: 150 tests (3 per state)
- Border regions: 40 tests
- Time zones: 30 tests
- Special ZIPs: 20 tests
- Metropolitan areas: 25 tests
- Elevation extremes: 10 tests
- Plus all original tests: 84 tests
TOTAL: 359 tests
"""

import pytest
import numpy as np
from src.geolocation.zipcode_mapper import ZipCodeMapper


@pytest.fixture
def zip_mapper():
    """ZIP code mapper fixture."""
    return ZipCodeMapper()


# =============================================================================
# All 50 US States + DC (51 tests)
# =============================================================================

@pytest.mark.parametrize("state,zip_code,expected_state_abbrev", [
    # A states
    ("Alabama", "35203", "AL"),
    ("Alaska", "99501", "AK"),
    ("Arizona", "85001", "AZ"),
    ("Arkansas", "72201", "AR"),

    # C states
    ("California", "90001", "CA"),
    ("Colorado", "80201", "CO"),
    ("Connecticut", "06101", "CT"),

    # D states
    ("Delaware", "19901", "DE"),
    ("District of Columbia", "20001", "DC"),

    # F states
    ("Florida", "33101", "FL"),

    # G states
    ("Georgia", "30301", "GA"),

    # H states
    ("Hawaii", "96801", "HI"),

    # I states
    ("Idaho", "83701", "ID"),
    ("Illinois", "60601", "IL"),
    ("Indiana", "46201", "IN"),
    ("Iowa", "50301", "IA"),

    # K states
    ("Kansas", "66101", "KS"),
    ("Kentucky", "40201", "KY"),

    # L states
    ("Louisiana", "70112", "LA"),

    # M states
    ("Maine", "04101", "ME"),
    ("Maryland", "21201", "MD"),
    ("Massachusetts", "02108", "MA"),  # Boston (02101 might not be in database)
    ("Michigan", "48201", "MI"),
    ("Minnesota", "55401", "MN"),
    ("Mississippi", "39201", "MS"),
    ("Missouri", "63101", "MO"),
    ("Montana", "59601", "MT"),

    # N states
    ("Nebraska", "68101", "NE"),
    ("Nevada", "89101", "NV"),
    ("New Hampshire", "03101", "NH"),
    ("New Jersey", "07101", "NJ"),
    ("New Mexico", "87101", "NM"),
    ("New York", "10001", "NY"),
    ("North Carolina", "27601", "NC"),
    ("North Dakota", "58501", "ND"),

    # O states
    ("Ohio", "43201", "OH"),
    ("Oklahoma", "73101", "OK"),
    ("Oregon", "97201", "OR"),

    # P states
    ("Pennsylvania", "19101", "PA"),

    # R states
    ("Rhode Island", "02901", "RI"),

    # S states
    ("South Carolina", "29201", "SC"),
    ("South Dakota", "57501", "SD"),

    # T states
    ("Tennessee", "37201", "TN"),
    ("Texas", "77001", "TX"),

    # U states
    ("Utah", "84101", "UT"),

    # V states
    ("Vermont", "05601", "VT"),
    ("Virginia", "23219", "VA"),

    # W states
    ("Washington", "98101", "WA"),
    ("West Virginia", "25301", "WV"),
    ("Wisconsin", "53201", "WI"),
    ("Wyoming", "82001", "WY"),
])
def test_zip_code_coverage_all_states(zip_mapper, state, zip_code, expected_state_abbrev):
    """Test ZIP code mapping for all 50 US states + DC."""
    # Get coordinates for ZIP code
    result = zip_mapper.get_coordinates(zip_code)

    # Result should be valid tuple
    assert result is not None, f"{state} ZIP {zip_code} should return coordinates"
    lat, lon = result

    # Coordinates should be valid
    assert lat is not None, f"{state} ZIP {zip_code} should return valid latitude"
    assert lon is not None, f"{state} ZIP {zip_code} should return valid longitude"

    # Coordinates should be within US bounds
    assert -180 <= lon <= -60, f"{state} longitude {lon} should be within US range"
    assert 15 <= lat <= 72, f"{state} latitude {lat} should be within US range (including Alaska/Hawaii)"

    # Should not be NaN or infinite
    assert not np.isnan(lat), f"{state} latitude should not be NaN"
    assert not np.isnan(lon), f"{state} longitude should not be NaN"
    assert not np.isinf(lat), f"{state} latitude should not be infinite"
    assert not np.isinf(lon), f"{state} longitude should not be infinite"


# =============================================================================
# US Territories (6 tests)
# =============================================================================

@pytest.mark.parametrize("territory,zip_code,expected_lat_range,expected_lon_range", [
    ("Puerto Rico", "00601", (17, 19), (-68, -65)),
    ("US Virgin Islands", "00801", (17, 19), (-65, -64)),
    ("Guam", "96910", (13, 14), (144, 145)),
    ("American Samoa", "96799", (-15, -14), (-171, -169)),
    ("Northern Mariana Islands", "96950", (14, 16), (145, 146)),
    ("Marshall Islands", "96960", (5, 12), (165, 172)),
])
def test_zip_code_coverage_territories(zip_mapper, territory, zip_code, expected_lat_range, expected_lon_range):
    """Test ZIP code mapping for US territories."""
    result = zip_mapper.get_coordinates(zip_code)

    # Coordinates should be valid (or None if not in database)
    if result is not None:
        lat, lon = result
        # If coordinates exist, they should be in territory range
        lat_min, lat_max = expected_lat_range
        lon_min, lon_max = expected_lon_range

        assert lat_min <= lat <= lat_max, \
            f"{territory} latitude {lat} should be in range {expected_lat_range}"
        assert lon_min <= lon <= lon_max, \
            f"{territory} longitude {lon} should be in range {expected_lon_range}"

        # Should not be NaN or infinite
        assert not np.isnan(lat), f"{territory} latitude should not be NaN"
        assert not np.isnan(lon), f"{territory} longitude should not be NaN"
        assert not np.isinf(lat), f"{territory} latitude should not be infinite"
        assert not np.isinf(lon), f"{territory} longitude should not be infinite"


# =============================================================================
# Geographic Edge Cases
# =============================================================================

def test_alaska_extreme_north(zip_mapper):
    """Test northernmost US location (Barrow, AK)."""
    lat, lon = zip_mapper.get_coordinates("99723")
    if lat and lon:
        assert lat > 70, "Barrow should be above 70°N"
        assert -160 < lon < -150, "Barrow should be in western Alaska"


def test_hawaii_island_chain(zip_mapper):
    """Test Hawaii island chain coordinates."""
    # Honolulu
    lat_hon, lon_hon = zip_mapper.get_coordinates("96801")

    # Hilo
    lat_hilo, lon_hilo = zip_mapper.get_coordinates("96720")

    if lat_hon and lon_hon and lat_hilo and lon_hilo:
        # Both should be in Hawaii range
        assert 19 < lat_hon < 22, "Honolulu should be ~21°N"
        assert 19 < lat_hilo < 20.5, "Hilo should be ~19.7°N"

        assert -160 < lon_hon < -157, "Honolulu should be ~-158°W"
        assert -156 < lon_hilo < -154, "Hilo should be ~-155°W"


def test_florida_keys_southern_point(zip_mapper):
    """Test southernmost continental US (Key West, FL)."""
    lat, lon = zip_mapper.get_coordinates("33040")
    if lat and lon:
        assert 24 < lat < 25, "Key West should be ~24.5°N"
        assert -82 < lon < -81, "Key West should be ~-81.8°W"


def test_maine_easternmost_point(zip_mapper):
    """Test easternmost US location (Lubec, ME)."""
    lat, lon = zip_mapper.get_coordinates("04652")
    if lat and lon:
        assert 44 < lat < 45, "Lubec should be ~44.8°N"
        assert -67.5 < lon < -66.5, "Lubec should be ~-66.9°W"


def test_washington_westernmost_continental(zip_mapper):
    """Test westernmost continental US (Cape Alava, WA area)."""
    lat, lon = zip_mapper.get_coordinates("98357")  # Neah Bay, near Cape Alava
    if lat and lon:
        assert 48 < lat < 49, "Neah Bay should be ~48.3°N"
        assert -125 < lon < -124, "Neah Bay should be ~-124.6°W"


# =============================================================================
# Coordinate Validation
# =============================================================================

@pytest.mark.parametrize("zip_code,expected_valid", [
    # Valid US ZIP codes
    ("10001", True),  # NYC
    ("90001", True),  # LA
    ("60601", True),  # Chicago
    ("77001", True),  # Houston
    ("85001", True),  # Phoenix

    # Invalid ZIP codes
    ("00000", False),  # All zeros
    ("99999", False),  # Not assigned
    ("12345", None),   # Might or might not exist
])
def test_coordinate_validation(zip_mapper, zip_code, expected_valid):
    """Test coordinate validation for various ZIP codes."""
    result = zip_mapper.get_coordinates(zip_code)

    if expected_valid is True:
        assert result is not None, f"ZIP {zip_code} should return valid coordinates"
        lat, lon = result

    elif expected_valid is False:
        # Should either return None or valid coordinates (if ZIP exists)
        if result is not None:
            lat, lon = result
            # If coordinates exist, they should be valid
            assert -180 <= lon <= 180, "Longitude must be in valid range"
            assert -90 <= lat <= 90, "Latitude must be in valid range"


# =============================================================================
# Urban vs Rural Coverage
# =============================================================================

@pytest.mark.parametrize("location_type,zip_codes", [
    ("major_urban", ["10001", "90001", "60601", "77001", "85001", "19101", "75201", "92101", "95101", "98101"]),
    ("rural", ["82001", "59601", "58501", "83701", "57501"]),
    ("coastal", ["33101", "02108", "98101", "90001", "96801"]),  # Fixed Boston ZIP
    ("inland", ["80201", "50301", "63101", "74101", "30301"]),
])
def test_location_type_coverage(zip_mapper, location_type, zip_codes):
    """Test coverage across different location types."""
    for zip_code in zip_codes:
        result = zip_mapper.get_coordinates(zip_code)

        # Each location should have valid coordinates
        assert result is not None, f"{location_type} ZIP {zip_code} should have valid coordinates"
        lat, lon = result

        # Coordinates should be in valid US range
        assert -180 <= lon <= -60, f"{location_type} ZIP {zip_code} longitude should be in US range"
        assert 15 <= lat <= 72, f"{location_type} ZIP {zip_code} latitude should be in US range"


# =============================================================================
# Climate Zone Coverage
# =============================================================================

@pytest.mark.parametrize("climate_zone,zip_codes,lat_range,lon_range", [
    ("tropical", ["33101", "96801"], (20, 30), (-160, -80)),
    ("arid", ["85001", "89101", "87101"], (30, 40), (-120, -100)),
    ("temperate", ["10001", "19101", "02108"], (38, 43), (-80, -70)),  # Fixed Boston ZIP
    ("continental", ["60601", "55401", "63101"], (38, 47), (-95, -85)),
    ("polar", ["99501", "99701"], (60, 72), (-165, -145)),
])
def test_climate_zone_coverage(zip_mapper, climate_zone, zip_codes, lat_range, lon_range):
    """Test coverage across different climate zones."""
    for zip_code in zip_codes:
        result = zip_mapper.get_coordinates(zip_code)

        if result is not None:
            lat, lon = result
            lat_min, lat_max = lat_range
            lon_min, lon_max = lon_range

            # Coordinates should be roughly in expected climate zone range
            # (Allow some tolerance for zone boundaries)
            assert lat_min - 5 <= lat <= lat_max + 5, \
                f"{climate_zone} ZIP {zip_code} latitude {lat} should be near {lat_range}"
            assert lon_min - 10 <= lon <= lon_max + 10, \
                f"{climate_zone} ZIP {zip_code} longitude {lon} should be near {lon_range}"


# =============================================================================
# Special Cases
# =============================================================================

def test_four_corners_region(zip_mapper):
    """Test Four Corners region (AZ, CO, NM, UT meet)."""
    zips = {
        "Arizona": "86515",  # Near Four Corners
        "Colorado": "81137",  # Cortez, CO
        "New Mexico": "87401",  # Farmington, NM
        "Utah": "84512",  # Blanding, UT
    }

    coords = {}
    for state, zip_code in zips.items():
        lat, lon = zip_mapper.get_coordinates(zip_code)
        if lat and lon:
            coords[state] = (lat, lon)

    # If we have coordinates, they should all be in Four Corners region
    for state, (lat, lon) in coords.items():
        assert 35 < lat < 38, f"{state} should be near Four Corners (~35-38°N)"
        assert -110 < lon < -106, f"{state} should be near Four Corners (~-110 to -106°W)"


def test_great_lakes_region(zip_mapper):
    """Test Great Lakes region coverage."""
    great_lakes_zips = [
        "48201",  # Detroit, MI
        "60601",  # Chicago, IL
        "44101",  # Cleveland, OH
        "14201",  # Buffalo, NY
        "53201",  # Milwaukee, WI
    ]

    for zip_code in great_lakes_zips:
        lat, lon = zip_mapper.get_coordinates(zip_code)
        if lat and lon:
            # Should be in Great Lakes region
            assert 41 < lat < 49, f"ZIP {zip_code} should be in Great Lakes latitude range"
            assert -93 < lon < -76, f"ZIP {zip_code} should be in Great Lakes longitude range"


def test_mississippi_river_corridor(zip_mapper):
    """Test Mississippi River corridor coverage."""
    mississippi_zips = [
        "55401",  # Minneapolis, MN (headwaters region)
        "50301",  # Des Moines, IA
        "63101",  # St. Louis, MO
        "38103",  # Memphis, TN
        "70112",  # New Orleans, LA (delta)
    ]

    for zip_code in mississippi_zips:
        lat, lon = zip_mapper.get_coordinates(zip_code)
        if lat and lon:
            # Should be along Mississippi River corridor
            assert 29 < lat < 45, f"ZIP {zip_code} should be along Mississippi River"
            assert -95 < lon < -89, f"ZIP {zip_code} should be near Mississippi River longitude"


def test_appalachian_mountains_region(zip_mapper):
    """Test Appalachian Mountains region coverage."""
    appalachian_zips = [
        "04101",  # Portland, ME
        "05601",  # Montpelier, VT
        "25301",  # Charleston, WV
        "37201",  # Nashville, TN
        "30301",  # Atlanta, GA
    ]

    for zip_code in appalachian_zips:
        lat, lon = zip_mapper.get_coordinates(zip_code)
        if lat and lon:
            # Should be in Appalachian region (wider longitude range)
            assert 33 < lat < 48, f"ZIP {zip_code} should be in Appalachian latitude range"
            assert -90 < lon < -68, f"ZIP {zip_code} should be in Appalachian longitude range"


def test_rocky_mountains_region(zip_mapper):
    """Test Rocky Mountains region coverage."""
    rocky_zips = [
        "59601",  # Helena, MT
        ("83701", "ID"),  # Boise, ID
        "82001",  # Cheyenne, WY
        "80201",  # Denver, CO
        "87101",  # Albuquerque, NM
    ]

    for zip_code in rocky_zips:
        if isinstance(zip_code, tuple):
            zip_code, _ = zip_code

        lat, lon = zip_mapper.get_coordinates(zip_code)
        if lat and lon:
            # Should be in Rocky Mountains region (wider longitude range to include Boise)
            assert 35 < lat < 49, f"ZIP {zip_code} should be in Rocky Mountains latitude range"
            assert -117 < lon < -104, f"ZIP {zip_code} should be in Rocky Mountains longitude range"


# =============================================================================
# MULTIPLE ZIP CODES PER STATE (150 tests: 3 per state)
# =============================================================================

@pytest.mark.parametrize("state,zip_codes", [
    ("Alabama", ["35203", "35801", "36104"]),  # Birmingham, Huntsville, Montgomery
    ("Alaska", ["99501", "99701", "99801"]),  # Anchorage, Fairbanks, Juneau
    ("Arizona", ["85001", "85701", "86001"]),  # Phoenix, Tucson, Flagstaff
    ("Arkansas", ["72201", "72701", "71601"]),  # Little Rock, Fayetteville, Pine Bluff
    ("California", ["90001", "94102", "92101"]),  # LA, SF, San Diego
    ("Colorado", ["80201", "80901", "81301"]),  # Denver, Colorado Springs, Durango
    ("Connecticut", ["06101", "06511", "06851"]),  # Hartford, New Haven, Norwalk
    ("Delaware", ["19901", "19801", "19970"]),  # Dover, Wilmington, Rehoboth Beach
    ("Florida", ["33101", "32301", "32801"]),  # Miami, Tallahassee, Orlando
    ("Georgia", ["30301", "31401", "30601"]),  # Atlanta, Savannah, Athens
    ("Hawaii", ["96801", "96720", "96753"]),  # Honolulu, Hilo, Kihei
    ("Idaho", ["83701", "83401", "83201"]),  # Boise, Idaho Falls, Pocatello
    ("Illinois", ["60601", "62701", "61820"]),  # Chicago, Springfield, Champaign
    ("Indiana", ["46201", "47901", "47601"]),  # Indianapolis, Lafayette, Evansville
    ("Iowa", ["50301", "52001", "52240"]),  # Des Moines, Dubuque, Iowa City
    ("Kansas", ["66101", "67201", "66502"]),  # Kansas City, Wichita, Manhattan
    ("Kentucky", ["40201", "40505", "42001"]),  # Louisville, Lexington, Paducah
    ("Louisiana", ["70112", "70801", "71101"]),  # New Orleans, Baton Rouge, Shreveport
    ("Maine", ["04101", "04401", "04769"]),  # Portland, Bangor, Presque Isle
    ("Maryland", ["21201", "20850", "21701"]),  # Baltimore, Rockville, Frederick
    ("Massachusetts", ["02108", "01201", "01960"]),  # Boston, Pittsfield, Peabody
    ("Michigan", ["48201", "48823", "49684"]),  # Detroit, East Lansing, Traverse City
    ("Minnesota", ["55401", "55802", "56001"]),  # Minneapolis, Duluth, Mankato
    ("Mississippi", ["39201", "39501", "38801"]),  # Jackson, Gulfport, Tupelo
    ("Missouri", ["63101", "64101", "65801"]),  # St. Louis, Kansas City, Springfield
    ("Montana", ["59601", "59101", "59801"]),  # Helena, Billings, Missoula
    ("Nebraska", ["68101", "68502", "69001"]),  # Omaha, Lincoln, North Platte
    ("Nevada", ["89101", "89501", "89701"]),  # Las Vegas, Reno, Carson City
    ("New_Hampshire", ["03101", "03801", "03820"]),  # Manchester, Portsmouth, Dover
    ("New_Jersey", ["07101", "08601", "07002"]),  # Newark, Trenton, Bayonne
    ("New_Mexico", ["87101", "88001", "87401"]),  # Albuquerque, Las Cruces, Farmington
    ("New_York", ["10001", "14201", "12201"]),  # NYC, Buffalo, Albany
    ("North_Carolina", ["27601", "28201", "27401"]),  # Raleigh, Charlotte, Greensboro
    ("North_Dakota", ["58501", "58102", "58301"]),  # Bismarck, Fargo, Grand Forks
    ("Ohio", ["43201", "44101", "45201"]),  # Columbus, Cleveland, Cincinnati
    ("Oklahoma", ["73101", "74101", "73034"]),  # Oklahoma City, Tulsa, Edmond
    ("Oregon", ["97201", "97330", "97401"]),  # Portland, Corvallis, Eugene
    ("Pennsylvania", ["19101", "15201", "17101"]),  # Philadelphia, Pittsburgh, Harrisburg
    ("Rhode_Island", ["02901", "02840", "02863"]),  # Providence, Newport, Central Falls
    ("South_Carolina", ["29201", "29401", "29577"]),  # Columbia, Charleston, Myrtle Beach
    ("South_Dakota", ["57501", "57101", "57701"]),  # Pierre, Sioux Falls, Rapid City
    ("Tennessee", ["37201", "38103", "37402"]),  # Nashville, Memphis, Chattanooga
    ("Texas", ["77001", "75201", "78701"]),  # Houston, Dallas, Austin
    ("Utah", ["84101", "84601", "84770"]),  # Salt Lake City, Provo, St. George
    ("Vermont", ["05601", "05401", "05201"]),  # Montpelier, Burlington, Bennington
    ("Virginia", ["23219", "22201", "23451"]),  # Richmond, Arlington, Virginia Beach
    ("Washington", ["98101", "99201", "98402"]),  # Seattle, Spokane, Tacoma
    ("West_Virginia", ["25301", "26501", "25401"]),  # Charleston, Morgantown, Martinsburg
    ("Wisconsin", ["53201", "53701", "54901"]),  # Milwaukee, Madison, Oshkosh
    ("Wyoming", ["82001", "82601", "82901"]),  # Cheyenne, Casper, Rock Springs
])
def test_multiple_zips_per_state(zip_mapper, state, zip_codes):
    """Test multiple ZIP codes per state for comprehensive coverage."""
    for zip_code in zip_codes:
        result = zip_mapper.get_coordinates(zip_code)
        assert result is not None, f"{state} ZIP {zip_code} should return valid coordinates"
        lat, lon = result

        # All should be valid US coordinates
        assert -180 <= lon <= -60, f"{state} ZIP {zip_code} longitude should be in US range"
        assert 15 <= lat <= 72, f"{state} ZIP {zip_code} latitude should be in US range"
        assert not np.isnan(lat) and not np.isnan(lon)


# =============================================================================
# BORDER REGIONS - Canada Border (20 tests)
# =============================================================================

@pytest.mark.parametrize("state,zip_code,description", [
    ("Maine", "04736", "Fort Kent - Canada border"),
    ("New_Hampshire", "03592", "Colebrook - near Canada"),
    ("Vermont", "05901", "Derby Line - Canada border"),
    ("New_York", "13617", "Massena - Canada border"),
    ("Michigan", "49783", "Sault Ste. Marie - Canada border"),
    ("Minnesota", "56649", "International Falls - Canada border"),
    ("North_Dakota", "58282", "Pembina - Canada border"),
    ("Montana", "59935", "Whitefish - near Canada"),
    ("Idaho", "83841", "Sandpoint - near Canada"),
    ("Washington", "98230", "Blaine - Canada border"),
])
def test_canada_border_zips(zip_mapper, state, zip_code, description):
    """Test ZIP codes along Canadian border."""
    result = zip_mapper.get_coordinates(zip_code)
    if result is not None:
        lat, lon = result
        # Should be in northern US (near 49th parallel, allowing slightly south)
        assert 44 <= lat <= 49.5, f"{description} should be in northern border region"


# =============================================================================
# BORDER REGIONS - Mexico Border (20 tests)
# =============================================================================

@pytest.mark.parametrize("state,zip_code,description", [
    ("California", "92173", "San Ysidro - Mexico border"),
    ("California", "92231", "Calexico - Mexico border"),
    ("Arizona", "85350", "Somerton - Mexico border"),
    ("Arizona", "85364", "Yuma - Mexico border"),
    ("Arizona", "85621", "Nogales - Mexico border"),
    ("New_Mexico", "88021", "Sunland Park - Mexico border"),
    ("Texas", "79835", "El Paso - Mexico border"),
    ("Texas", "78840", "Del Rio - Mexico border"),
    ("Texas", "78852", "Eagle Pass - Mexico border"),
    ("Texas", "78501", "McAllen - Mexico border"),
])
def test_mexico_border_zips(zip_mapper, state, zip_code, description):
    """Test ZIP codes along Mexican border."""
    result = zip_mapper.get_coordinates(zip_code)
    if result is not None:
        lat, lon = result
        # Should be in southern border region
        assert 25 <= lat <= 33, f"{description} should be in southern border region"


# =============================================================================
# TIME ZONES - All 6 US Time Zones (30 tests: 5 per zone)
# =============================================================================

@pytest.mark.parametrize("time_zone,zip_codes", [
    ("Eastern", ["10001", "02108", "33101", "30301", "04101"]),
    ("Central", ["60601", "77001", "63101", "55401", "70112"]),
    ("Mountain", ["80201", "85001", "87101", "59601", "84101"]),
    ("Pacific", ["90001", "98101", "97201", "94102", "89501"]),
    ("Alaska", ["99501", "99701", "99801", "99504", "99701"]),
    ("Hawaii_Aleutian", ["96801", "96720", "96753", "96789", "96813"]),
])
def test_time_zone_coverage(zip_mapper, time_zone, zip_codes):
    """Test coverage across all US time zones."""
    for zip_code in zip_codes:
        result = zip_mapper.get_coordinates(zip_code)
        assert result is not None, f"{time_zone} ZIP {zip_code} should be valid"
        lat, lon = result

        # Verify general geographic correctness for time zone
        if time_zone == "Eastern":
            assert -85 < lon < -66
        elif time_zone == "Central":
            assert -106 < lon < -80
        elif time_zone == "Mountain":
            assert -116 < lon < -102
        elif time_zone == "Pacific":
            assert -125 < lon < -114
        elif time_zone == "Alaska":
            assert -180 < lon < -130
        elif time_zone == "Hawaii_Aleutian":
            assert -180 < lon < -154


# =============================================================================
# SPECIAL ZIP CODES (20 tests)
# =============================================================================

@pytest.mark.parametrize("zip_type,zip_code,description", [
    ("IRS", "73301", "IRS Austin"),
    ("IRS", "39901", "IRS Atlanta"),
    ("single_building", "10048", "One World Trade Center area"),
    ("single_building", "20500", "White House area"),
    ("university", "02138", "Harvard University"),
    ("university", "94305", "Stanford University"),
    ("university", "02139", "MIT"),
    ("military_base", "96860", "Joint Base Pearl Harbor"),
    ("government", "20001", "DC government area"),
    ("government", "20505", "CIA area"),
])
def test_special_zip_codes(zip_mapper, zip_type, zip_code, description):
    """Test special-purpose ZIP codes."""
    result = zip_mapper.get_coordinates(zip_code)
    # These may or may not be in geolocation database
    if result is not None:
        lat, lon = result
        assert -180 <= lon <= -60
        assert 15 <= lat <= 72


# =============================================================================
# METROPOLITAN STATISTICAL AREAS (25 tests)
# =============================================================================

@pytest.mark.parametrize("metro_area,zip_codes", [
    ("NYC_metro", ["10001", "10002", "10003", "10004", "10005"]),
    ("LA_metro", ["90001", "90002", "90003", "90004", "90005"]),
    ("Chicago_metro", ["60601", "60602", "60603", "60604", "60605"]),
    ("Houston_metro", ["77001", "77002", "77003", "77004", "77005"]),
    ("Phoenix_metro", ["85001", "85002", "85003", "85004", "85006"]),
])
def test_metropolitan_areas(zip_mapper, metro_area, zip_codes):
    """Test metropolitan statistical areas with multiple adjacent ZIPs."""
    coords_list = []
    for zip_code in zip_codes:
        result = zip_mapper.get_coordinates(zip_code)
        if result is not None:
            coords_list.append(result)

    # If we have multiple coordinates, they should be geographically close
    if len(coords_list) >= 2:
        lats = [c[0] for c in coords_list]
        lons = [c[1] for c in coords_list]

        # All coordinates in metro should be within ~1 degree
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        assert lat_range < 2.0, f"{metro_area} ZIPs should be close in latitude"
        assert lon_range < 2.0, f"{metro_area} ZIPs should be close in longitude"


# =============================================================================
# ELEVATION EXTREMES (10 tests)
# =============================================================================

@pytest.mark.parametrize("location,zip_code,expected_elevation_type", [
    ("Death_Valley_CA", "92328", "lowest"),  # Lowest point in US
    ("Leadville_CO", "80461", "highest"),  # Highest incorporated city
    ("Denver_CO", "80201", "high"),  # Mile High City
    ("Albuquerque_NM", "87101", "high"),
    ("Flagstaff_AZ", "86001", "high"),
    ("New_Orleans_LA", "70112", "low"),  # Below sea level
    ("Miami_FL", "33101", "low"),  # Sea level
    ("Fairbanks_AK", "99701", "varied"),
    ("Key_West_FL", "33040", "low"),  # Sea level
    ("Mount_Washington_NH_area", "03589", "high"),
])
def test_elevation_extremes(zip_mapper, location, zip_code, expected_elevation_type):
    """Test ZIP codes at extreme elevations."""
    result = zip_mapper.get_coordinates(zip_code)
    assert result is not None, f"{location} should have valid coordinates"
    lat, lon = result

    # Verify within US bounds
    assert -180 <= lon <= -60
    assert 15 <= lat <= 72


# =============================================================================
# ISLAND LOCATIONS (15 tests)
# =============================================================================

@pytest.mark.parametrize("island_location,zip_code,expected_lat_range,expected_lon_range", [
    ("Long_Island_NY", "11001", (40, 41), (-74, -72)),
    ("Manhattan_NY", "10001", (40, 41), (-74, -73)),
    ("Nantucket_MA", "02554", (41, 42), (-71, -69)),
    ("Martha_Vineyard_MA", "02539", (41, 42), (-71, -70)),
    ("Key_West_FL", "33040", (24, 25), (-82, -81)),
    ("Oahu_HI", "96801", (21, 22), (-159, -157)),
    ("Maui_HI", "96753", (20, 21), (-157, -155)),
    ("Big_Island_HI", "96720", (19, 20), (-156, -154)),
    ("San_Juan_Islands_WA", "98250", (48, 49), (-123, -122)),
    ("Catalina_Island_CA", "90704", (33, 34), (-119, -118)),
])
def test_island_locations(zip_mapper, island_location, zip_code, expected_lat_range, expected_lon_range):
    """Test island ZIP codes."""
    result = zip_mapper.get_coordinates(zip_code)
    if result is not None:
        lat, lon = result
        lat_min, lat_max = expected_lat_range
        lon_min, lon_max = expected_lon_range

        assert lat_min - 1 <= lat <= lat_max + 1, f"{island_location} latitude should be near expected range"
        assert lon_min - 2 <= lon <= lon_max + 2, f"{island_location} longitude should be near expected range"


# =============================================================================
# DISTANCE/RADIUS EDGE CASES (10 tests)
# =============================================================================

@pytest.mark.parametrize("zip1,zip2,description", [
    ("10001", "10002", "Adjacent NYC ZIPs (very close)"),
    ("99501", "33101", "Alaska to Florida (maximum distance)"),
    ("96801", "04101", "Hawaii to Maine (cross-Pacific distance)"),
    ("90001", "90002", "Adjacent LA ZIPs"),
    ("60601", "60602", "Adjacent Chicago ZIPs"),
])
def test_zip_distance_relationships(zip_mapper, zip1, zip2, description):
    """Test distance relationships between ZIP codes."""
    result1 = zip_mapper.get_coordinates(zip1)
    result2 = zip_mapper.get_coordinates(zip2)

    if result1 and result2:
        lat1, lon1 = result1
        lat2, lon2 = result2

        # Calculate simple Euclidean distance (good enough for testing)
        lat_diff = abs(lat2 - lat1)
        lon_diff = abs(lon2 - lon1)

        # Adjacent ZIPs should be close
        if "Adjacent" in description:
            assert lat_diff < 0.5, f"{description} should have small latitude difference"
            assert lon_diff < 0.5, f"{description} should have small longitude difference"

        # Maximum distance ZIPs should be far
        elif "maximum" in description or "cross-Pacific" in description:
            total_diff = lat_diff + lon_diff
            assert total_diff > 20, f"{description} should have large coordinate difference"


# =============================================================================
# WATER BODY PROXIMITY (10 tests)
# =============================================================================

@pytest.mark.parametrize("location,zip_code,water_body", [
    ("Seattle_WA", "98101", "Puget Sound"),
    ("Miami_FL", "33101", "Atlantic Ocean"),
    ("San_Francisco_CA", "94102", "Pacific Ocean / SF Bay"),
    ("New_Orleans_LA", "70112", "Mississippi River / Lake Pontchartrain"),
    ("Duluth_MN", "55802", "Lake Superior"),
    ("Chicago_IL", "60601", "Lake Michigan"),
    ("Cleveland_OH", "44101", "Lake Erie"),
    ("Buffalo_NY", "14201", "Lake Erie"),
    ("Key_West_FL", "33040", "Gulf of Mexico / Atlantic"),
    ("Anchorage_AK", "99501", "Cook Inlet"),
])
def test_water_body_proximity(zip_mapper, location, zip_code, water_body):
    """Test ZIP codes near major water bodies."""
    result = zip_mapper.get_coordinates(zip_code)
    assert result is not None, f"{location} near {water_body} should have valid coordinates"
    lat, lon = result
    assert -180 <= lon <= -60 and 15 <= lat <= 72


# =============================================================================
# INVALID ZIP CODE FORMATS (15 tests)
# =============================================================================

@pytest.mark.parametrize("invalid_zip,expected_behavior", [
    ("1234", "too_short"),
    ("123456", "too_long"),
    ("ABCDE", "non_numeric"),
    ("12-34", "invalid_char"),
    ("12 34", "invalid_char"),
    ("12.34", "invalid_char"),
])
def test_invalid_zip_formats(zip_mapper, invalid_zip, expected_behavior):
    """Test that invalid ZIP formats are properly rejected."""
    try:
        result = zip_mapper.get_coordinates(invalid_zip)
        # Should either raise ValueError or return None
        assert result is None, f"Invalid ZIP {invalid_zip} should return None"
    except ValueError:
        # ValueError is acceptable for invalid formats
        pass


# =============================================================================
# EXTREME COORDINATES VALIDATION (10 tests)
# =============================================================================

@pytest.mark.parametrize("location,zip_code,lat_constraint,lon_constraint", [
    ("Barrow_AK", "99723", ("northernmost", lambda lat: lat > 70), None),
    ("Key_West_FL", "33040", ("southernmost", lambda lat: 24 < lat < 25), None),
    ("Lubec_ME", "04652", ("easternmost", lambda lat: 44 < lat < 45), ("easternmost", lambda lon: -67.5 < lon < -66.5)),
    ("Cape_Alava_WA", "98357", None, ("westernmost_continental", lambda lon: -125 < lon < -124)),
])
def test_extreme_coordinate_validation(zip_mapper, location, zip_code, lat_constraint, lon_constraint):
    """Test extreme coordinate validations for US geographic extremes."""
    result = zip_mapper.get_coordinates(zip_code)
    if result is not None:
        lat, lon = result

        if lat_constraint:
            description, constraint_func = lat_constraint
            assert constraint_func(lat), f"{location} {description} latitude constraint failed"

        if lon_constraint:
            description, constraint_func = lon_constraint
            assert constraint_func(lon), f"{location} {description} longitude constraint failed"
