"""
Comprehensive tests for ZIP Code Mapper.

NO MOCKS - Testing with REAL pgeocode library and actual ZIP code data.
Target: 95% coverage for input validation and location lookup logic.

Total test cases: ~30
"""

import pytest
from src.geolocation.zipcode_mapper import ZipCodeMapper


class TestGetCoordinates:
    """Test get_coordinates method with real pgeocode data."""

    def setup_method(self):
        self.mapper = ZipCodeMapper()

    def test_valid_zip_washington_dc(self):
        """Washington DC ZIP 20001 should return valid coordinates."""
        coords = self.mapper.get_coordinates('20001')
        assert coords is not None
        lat, lng = coords
        # DC is approximately 38.9°N, 77.0°W
        assert 38.5 < lat < 39.5
        assert -77.5 < lng < -76.5

    def test_valid_zip_new_york(self):
        """New York City ZIP 10001 should return valid coordinates."""
        coords = self.mapper.get_coordinates('10001')
        assert coords is not None
        lat, lng = coords
        # NYC is approximately 40.75°N, 74.0°W
        assert 40.0 < lat < 41.0
        assert -74.5 < lng < -73.5

    def test_valid_zip_anchorage(self):
        """Anchorage AK ZIP 99501 should return valid coordinates."""
        coords = self.mapper.get_coordinates('99501')
        assert coords is not None
        lat, lng = coords
        # Anchorage is approximately 61.2°N, 149.9°W
        assert 60.0 < lat < 62.0
        assert -151.0 < lng < -149.0

    def test_valid_zip_holtsville_leading_zeros(self):
        """ZIP 00501 (Holtsville, NY) with leading zeros should work."""
        coords = self.mapper.get_coordinates('00501')
        assert coords is not None
        lat, lng = coords
        # Holtsville is in Long Island, NY
        assert 40.0 < lat < 41.0
        assert -73.5 < lng < -72.5

    def test_invalid_format_too_short(self):
        """ZIP code with < 5 digits should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ZIP code length"):
            self.mapper.get_coordinates('123')

    def test_invalid_format_too_long(self):
        """ZIP code with > 5 digits should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ZIP code length"):
            self.mapper.get_coordinates('123456')

    def test_invalid_format_letters(self):
        """ZIP code with letters should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ZIP code format"):
            self.mapper.get_coordinates('ABCDE')

    def test_invalid_format_mixed_alphanumeric(self):
        """ZIP code with mixed letters and numbers should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ZIP code format"):
            self.mapper.get_coordinates('12A45')

    def test_invalid_format_with_space(self):
        """ZIP code with spaces should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ZIP code format"):
            self.mapper.get_coordinates('12 345')

    def test_valid_format_but_nonexistent_zip(self):
        """ZIP 00000 (valid format but doesn't exist) should return None."""
        coords = self.mapper.get_coordinates('00000')
        assert coords is None

    def test_strip_whitespace(self):
        """ZIP code with leading/trailing whitespace should work."""
        coords = self.mapper.get_coordinates(' 20001 ')
        assert coords is not None
        lat, lng = coords
        assert 38.5 < lat < 39.5

    def test_coordinates_are_floats(self):
        """Returned coordinates should be float types."""
        coords = self.mapper.get_coordinates('20001')
        assert coords is not None
        lat, lng = coords
        assert isinstance(lat, float)
        assert isinstance(lng, float)


class TestGetLocationInfo:
    """Test get_location_info method with real pgeocode data."""

    def setup_method(self):
        self.mapper = ZipCodeMapper()

    def test_location_info_washington_dc(self):
        """Get full location info for DC ZIP 20001."""
        info = self.mapper.get_location_info('20001')
        assert info is not None
        assert info['zip_code'] == '20001'
        assert info['latitude'] is not None
        assert info['longitude'] is not None
        assert info['state_code'] == 'DC'
        # Check that place_name exists and contains Washington
        assert info['place_name'] is not None

    def test_location_info_new_york(self):
        """Get full location info for NYC ZIP 10001."""
        info = self.mapper.get_location_info('10001')
        assert info is not None
        assert info['zip_code'] == '10001'
        assert info['state_code'] == 'NY'
        assert info['latitude'] is not None
        assert info['longitude'] is not None

    def test_location_info_structure(self):
        """Location info should have expected keys."""
        info = self.mapper.get_location_info('20001')
        assert info is not None
        required_keys = ['zip_code', 'latitude', 'longitude', 'place_name', 'state_code', 'state_name']
        for key in required_keys:
            assert key in info

    def test_location_info_invalid_format(self):
        """Invalid ZIP format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ZIP code"):
            self.mapper.get_location_info('ABCDE')

    def test_location_info_nonexistent_zip(self):
        """Non-existent ZIP should return None."""
        info = self.mapper.get_location_info('00000')
        assert info is None


class TestIsValidZipcode:
    """Test is_valid_zipcode validation method."""

    def setup_method(self):
        self.mapper = ZipCodeMapper()

    def test_valid_existing_zip(self):
        """Existing ZIP codes should be valid."""
        assert self.mapper.is_valid_zipcode('20001') is True
        assert self.mapper.is_valid_zipcode('10001') is True
        assert self.mapper.is_valid_zipcode('99501') is True

    def test_invalid_format_too_short(self):
        """ZIP with < 5 digits should be invalid."""
        assert self.mapper.is_valid_zipcode('123') is False
        assert self.mapper.is_valid_zipcode('1234') is False

    def test_invalid_format_too_long(self):
        """ZIP with > 5 digits should be invalid."""
        assert self.mapper.is_valid_zipcode('123456') is False
        assert self.mapper.is_valid_zipcode('1234567') is False

    def test_invalid_format_letters(self):
        """ZIP with letters should be invalid."""
        assert self.mapper.is_valid_zipcode('ABCDE') is False
        assert self.mapper.is_valid_zipcode('12ABC') is False

    def test_invalid_nonexistent_zip(self):
        """Non-existent ZIP (valid format) should be invalid."""
        assert self.mapper.is_valid_zipcode('00000') is False
        assert self.mapper.is_valid_zipcode('99999') is False

    def test_valid_with_whitespace(self):
        """Valid ZIP with whitespace should be valid after stripping."""
        assert self.mapper.is_valid_zipcode(' 20001 ') is True
        assert self.mapper.is_valid_zipcode('20001 ') is True
        assert self.mapper.is_valid_zipcode(' 20001') is True

    def test_invalid_empty_string(self):
        """Empty string should be invalid."""
        assert self.mapper.is_valid_zipcode('') is False

    def test_invalid_spaces_in_middle(self):
        """ZIP with spaces in middle should be invalid."""
        assert self.mapper.is_valid_zipcode('20 001') is False
        assert self.mapper.is_valid_zipcode('2 0001') is False


class TestCalculateDistance:
    """Test calculate_distance method with real ZIP codes."""

    def setup_method(self):
        self.mapper = ZipCodeMapper()

    def test_distance_same_zip(self):
        """Distance from ZIP to itself should be 0 miles."""
        distance = self.mapper.calculate_distance('20001', '20001')
        assert distance is not None
        assert distance == 0.0

    def test_distance_dc_to_nyc(self):
        """Distance from DC to NYC should be approximately 225 miles."""
        distance = self.mapper.calculate_distance('20001', '10001')
        assert distance is not None
        # DC to NYC is roughly 225 miles, allow 10% tolerance
        assert 200 < distance < 250

    def test_distance_nyc_to_la(self):
        """Distance from NYC to LA should be approximately 2450 miles."""
        distance = self.mapper.calculate_distance('10001', '90001')
        assert distance is not None
        # NYC to LA is roughly 2450 miles (straight line), allow 10% tolerance
        assert 2200 < distance < 2700

    def test_distance_is_symmetric(self):
        """Distance from A to B should equal distance from B to A."""
        dist_ab = self.mapper.calculate_distance('20001', '10001')
        dist_ba = self.mapper.calculate_distance('10001', '20001')
        assert dist_ab is not None
        assert dist_ba is not None
        # Should be equal within floating point precision
        assert abs(dist_ab - dist_ba) < 0.01

    def test_distance_invalid_first_zip(self):
        """Distance with invalid first ZIP should return None."""
        distance = self.mapper.calculate_distance('00000', '20001')
        assert distance is None

    def test_distance_invalid_second_zip(self):
        """Distance with invalid second ZIP should return None."""
        distance = self.mapper.calculate_distance('20001', '00000')
        assert distance is None

    def test_distance_both_invalid(self):
        """Distance with both ZIPs invalid should return None."""
        distance = self.mapper.calculate_distance('00000', '99999')
        assert distance is None

    def test_distance_returns_float(self):
        """Distance calculation should return float."""
        distance = self.mapper.calculate_distance('20001', '10001')
        assert distance is not None
        assert isinstance(distance, float)

    def test_distance_positive_value(self):
        """Distance should always be non-negative."""
        distance = self.mapper.calculate_distance('20001', '10001')
        assert distance is not None
        assert distance >= 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.mapper = ZipCodeMapper()

    def test_integer_zip_code_converted_to_string(self):
        """Integer ZIP code should be converted to string."""
        coords = self.mapper.get_coordinates(20001)
        assert coords is not None
        # Should work after conversion to string

    def test_multiple_lookups_consistent(self):
        """Multiple lookups of same ZIP should return same coordinates."""
        coords1 = self.mapper.get_coordinates('20001')
        coords2 = self.mapper.get_coordinates('20001')
        assert coords1 == coords2

    def test_different_instances_same_results(self):
        """Different mapper instances should return same results."""
        mapper1 = ZipCodeMapper()
        mapper2 = ZipCodeMapper()
        coords1 = mapper1.get_coordinates('20001')
        coords2 = mapper2.get_coordinates('20001')
        assert coords1 == coords2


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
