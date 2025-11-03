"""
Water Quality Index (WQI) Calculator

This module implements the Water Quality Index calculation based on multiple
water quality parameters. The WQI provides a single number that expresses
overall water quality.

Based on the National Sanitation Foundation Water Quality Index (NSF-WQI)
and EPA water quality standards.

References:
- National Sanitation Foundation WQI
- EPA Water Quality Standards
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WQICalculator:
    """
    Calculate Water Quality Index from water quality parameters.

    WQI Scale:
    - 90-100: Excellent
    - 70-89:  Good
    - 50-69:  Fair
    - 25-49:  Poor
    - 0-24:   Very Poor
    """

    # Parameter weights based on NSF-WQI methodology
    PARAMETER_WEIGHTS = {
        'dissolved_oxygen': 0.17,
        'ph': 0.11,
        'temperature': 0.10,
        'turbidity': 0.08,
        'total_phosphate': 0.10,
        'nitrate': 0.10,
        'fecal_coliform': 0.16,
        'biochemical_oxygen_demand': 0.11,
        'total_solids': 0.07
    }

    # Ideal parameter ranges and quality curves
    PARAMETER_STANDARDS = {
        'ph': {
            'ideal': 7.0,
            'min_acceptable': 6.5,
            'max_acceptable': 8.5,
            'critical_min': 4.0,
            'critical_max': 10.0
        },
        'dissolved_oxygen': {
            'unit': 'mg/L',
            'excellent': 9.0,
            'good': 7.0,
            'fair': 5.0,
            'poor': 3.0,
            'critical': 1.0
        },
        'temperature': {
            'unit': 'Celsius',
            'ideal': 20.0,
            'max_acceptable': 30.0,
            'critical_max': 35.0
        },
        'turbidity': {
            'unit': 'NTU',
            'excellent': 5.0,
            'good': 25.0,
            'fair': 50.0,
            'poor': 100.0
        },
        'nitrate': {
            'unit': 'mg/L',
            'excellent': 1.0,
            'good': 5.0,
            'fair': 10.0,
            'poor': 20.0,
            'critical': 50.0  # EPA MCL
        },
        'specific_conductance': {
            'unit': 'µS/cm',
            'excellent': 500,
            'good': 1000,
            'fair': 1500,
            'poor': 2000
        }
    }

    @staticmethod
    def calculate_ph_score(ph: float) -> float:
        """
        Calculate quality score for pH (0-100).

        Args:
            ph: pH value

        Returns:
            Quality score (0-100)
        """
        if pd.isna(ph) or ph < 0 or ph > 14:
            return np.nan

        # pH quality curve based on NSF-WQI
        if 6.5 <= ph <= 7.5:
            return 100  # Ideal range
        elif (6.0 <= ph < 6.5) or (7.5 < ph <= 8.0):
            return 90  # Good
        elif (5.5 <= ph < 6.0) or (8.0 < ph <= 8.5):
            return 70  # Fair
        elif (5.0 <= ph < 5.5) or (8.5 < ph <= 9.0):
            return 50  # Poor
        else:
            return max(0, 100 - abs(ph - 7.0) * 20)  # Very poor

    @staticmethod
    def calculate_do_score(dissolved_oxygen: float, saturation: Optional[float] = None) -> float:
        """
        Calculate quality score for Dissolved Oxygen (0-100).

        Args:
            dissolved_oxygen: DO concentration in mg/L
            saturation: Optional % saturation value

        Returns:
            Quality score (0-100)
        """
        if pd.isna(dissolved_oxygen) or dissolved_oxygen < 0:
            return np.nan

        # If saturation is provided, use it
        if saturation is not None and not pd.isna(saturation):
            if saturation >= 90:
                return 100
            elif saturation >= 70:
                return 85
            elif saturation >= 50:
                return 60
            elif saturation >= 30:
                return 35
            else:
                return 15

        # Otherwise use absolute DO concentration
        if dissolved_oxygen >= 9.0:
            return 100
        elif dissolved_oxygen >= 7.0:
            return 85
        elif dissolved_oxygen >= 5.0:
            return 60
        elif dissolved_oxygen >= 3.0:
            return 35
        elif dissolved_oxygen >= 1.0:
            return 15
        else:
            return 5

    @staticmethod
    def calculate_temperature_score(temperature: float) -> float:
        """
        Calculate quality score for temperature (0-100).

        Args:
            temperature: Temperature in Celsius

        Returns:
            Quality score (0-100)
        """
        if pd.isna(temperature):
            return np.nan

        # Temperature deviation from ideal affects aquatic life
        ideal_temp = 20.0
        deviation = abs(temperature - ideal_temp)

        if deviation <= 5:
            return 100
        elif deviation <= 10:
            return 80
        elif deviation <= 15:
            return 60
        elif deviation <= 20:
            return 40
        else:
            return max(0, 100 - deviation * 3)

    @staticmethod
    def calculate_turbidity_score(turbidity: float) -> float:
        """
        Calculate quality score for turbidity (0-100).

        Args:
            turbidity: Turbidity in NTU

        Returns:
            Quality score (0-100)
        """
        if pd.isna(turbidity) or turbidity < 0:
            return np.nan

        # Lower turbidity is better
        if turbidity <= 5:
            return 100
        elif turbidity <= 25:
            return 80
        elif turbidity <= 50:
            return 60
        elif turbidity <= 100:
            return 40
        else:
            return max(0, 100 - turbidity * 0.5)

    @staticmethod
    def calculate_nitrate_score(nitrate: float) -> float:
        """
        Calculate quality score for nitrate (0-100).

        Args:
            nitrate: Nitrate concentration in mg/L

        Returns:
            Quality score (0-100)
        """
        if pd.isna(nitrate) or nitrate < 0:
            return np.nan

        # EPA MCL for nitrate is 10 mg/L (as N)
        if nitrate <= 1.0:
            return 100
        elif nitrate <= 5.0:
            return 85
        elif nitrate <= 10.0:  # EPA MCL
            return 70
        elif nitrate <= 20.0:
            return 40
        elif nitrate <= 50.0:
            return 15
        else:
            return 5

    @staticmethod
    def calculate_conductance_score(conductance: float) -> float:
        """
        Calculate quality score for specific conductance (0-100).

        Args:
            conductance: Specific conductance in µS/cm

        Returns:
            Quality score (0-100)
        """
        if pd.isna(conductance) or conductance < 0:
            return np.nan

        # Lower conductance generally indicates better quality
        if conductance <= 500:
            return 100
        elif conductance <= 1000:
            return 80
        elif conductance <= 1500:
            return 60
        elif conductance <= 2000:
            return 40
        else:
            return max(0, 100 - (conductance - 2000) * 0.02)

    def calculate_wqi(
        self,
        ph: Optional[float] = None,
        dissolved_oxygen: Optional[float] = None,
        temperature: Optional[float] = None,
        turbidity: Optional[float] = None,
        nitrate: Optional[float] = None,
        conductance: Optional[float] = None,
        **kwargs
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Calculate overall Water Quality Index from available parameters.

        Args:
            ph: pH value
            dissolved_oxygen: Dissolved oxygen in mg/L
            temperature: Temperature in Celsius
            turbidity: Turbidity in NTU
            nitrate: Nitrate concentration in mg/L
            conductance: Specific conductance in µS/cm
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (WQI score, parameter scores dict, classification)
        """
        # Calculate individual parameter scores
        scores = {}
        weights_used = {}

        if ph is not None and not pd.isna(ph):
            scores['ph'] = self.calculate_ph_score(ph)
            weights_used['ph'] = 0.20

        if dissolved_oxygen is not None and not pd.isna(dissolved_oxygen):
            scores['dissolved_oxygen'] = self.calculate_do_score(dissolved_oxygen)
            weights_used['dissolved_oxygen'] = 0.25

        if temperature is not None and not pd.isna(temperature):
            scores['temperature'] = self.calculate_temperature_score(temperature)
            weights_used['temperature'] = 0.15

        if turbidity is not None and not pd.isna(turbidity):
            scores['turbidity'] = self.calculate_turbidity_score(turbidity)
            weights_used['turbidity'] = 0.15

        if nitrate is not None and not pd.isna(nitrate):
            scores['nitrate'] = self.calculate_nitrate_score(nitrate)
            weights_used['nitrate'] = 0.15

        if conductance is not None and not pd.isna(conductance):
            scores['conductance'] = self.calculate_conductance_score(conductance)
            weights_used['conductance'] = 0.10

        if not scores:
            logger.warning("No valid parameters provided for WQI calculation")
            return np.nan, {}, "Unknown"

        # Normalize weights to sum to 1.0
        total_weight = sum(weights_used.values())
        if total_weight == 0:
            return np.nan, scores, "Unknown"

        normalized_weights = {k: v / total_weight for k, v in weights_used.items()}

        # Calculate weighted WQI
        wqi = sum(scores[param] * normalized_weights[param]
                  for param in scores if not pd.isna(scores[param]))

        # Classify water quality
        classification = self.classify_wqi(wqi)

        logger.info(f"Calculated WQI: {wqi:.2f} ({classification})")

        return round(wqi, 2), scores, classification

    @staticmethod
    def classify_wqi(wqi: float) -> str:
        """
        Classify water quality based on WQI score.

        Args:
            wqi: Water Quality Index score

        Returns:
            Classification string
        """
        if pd.isna(wqi):
            return "Unknown"
        elif wqi >= 90:
            return "Excellent"
        elif wqi >= 70:
            return "Good"
        elif wqi >= 50:
            return "Fair"
        elif wqi >= 25:
            return "Poor"
        else:
            return "Very Poor"

    @staticmethod
    def is_safe(wqi: float) -> bool:
        """
        Determine if water is considered safe based on WQI.

        Args:
            wqi: Water Quality Index score

        Returns:
            True if safe (WQI >= 70), False otherwise
        """
        return not pd.isna(wqi) and wqi >= 70


if __name__ == "__main__":
    # Example usage and testing
    calculator = WQICalculator()

    print("Water Quality Index Calculator Test\n" + "=" * 60)

    # Test Case 1: Excellent water quality
    print("\nTest Case 1: Excellent Water Quality")
    wqi1, scores1, class1 = calculator.calculate_wqi(
        ph=7.2,
        dissolved_oxygen=9.5,
        temperature=20.0,
        turbidity=3.0,
        nitrate=0.5,
        conductance=400
    )
    print(f"  WQI: {wqi1:.2f}")
    print(f"  Classification: {class1}")
    print(f"  Safe for use: {'Yes' if calculator.is_safe(wqi1) else 'No'}")
    print(f"  Parameter scores: {scores1}")

    # Test Case 2: Fair water quality
    print("\nTest Case 2: Fair Water Quality")
    wqi2, scores2, class2 = calculator.calculate_wqi(
        ph=6.0,
        dissolved_oxygen=5.5,
        temperature=28.0,
        turbidity=40.0,
        nitrate=8.0
    )
    print(f"  WQI: {wqi2:.2f}")
    print(f"  Classification: {class2}")
    print(f"  Safe for use: {'Yes' if calculator.is_safe(wqi2) else 'No'}")

    # Test Case 3: Poor water quality
    print("\nTest Case 3: Poor Water Quality")
    wqi3, scores3, class3 = calculator.calculate_wqi(
        ph=5.0,
        dissolved_oxygen=2.5,
        temperature=32.0,
        turbidity=120.0,
        nitrate=15.0
    )
    print(f"  WQI: {wqi3:.2f}")
    print(f"  Classification: {class3}")
    print(f"  Safe for use: {'Yes' if calculator.is_safe(wqi3) else 'No'}")

    # Test Case 4: Partial data
    print("\nTest Case 4: Partial Data (only pH and DO)")
    wqi4, scores4, class4 = calculator.calculate_wqi(
        ph=7.5,
        dissolved_oxygen=8.0
    )
    print(f"  WQI: {wqi4:.2f}")
    print(f"  Classification: {class4}")
    print(f"  Parameters used: {list(scores4.keys())}")

    print("\n" + "=" * 60)
    print("WQI Calculator test complete!")
