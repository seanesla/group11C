"""
ML Feature Definitions and Categorization

This module defines the complete set of 59 ML features used by the water quality
prediction models, organized into logical categories.
"""

from typing import Dict, List


def get_feature_categories() -> Dict[str, Dict[str, any]]:
    """
    Get complete categorization of all 59 ML features.

    Returns:
        Dictionary mapping category names to feature lists with metadata
    """
    return {
        "water_quality": {
            "name": "🧪 Water Quality Parameters",
            "description": "Core NSF-WQI measurements from water samples",
            "features": {
                "ph": "Acidity/alkalinity level (0-14 scale)",
                "dissolved_oxygen": "Oxygen content in water (mg/L)",
                "temperature": "Water temperature (°C)",
                "conductance": "Electrical conductivity (µS/cm)",
                "nitrate": "Nitrate concentration (mg/L as N)"
            },
            "source": "Direct measurements",
            "available_for_us": True
        },

        "temporal": {
            "name": "📅 Temporal Features",
            "description": "Time-based features capturing trends and periods",
            "features": {
                "year": "Sample collection year",
                "years_since_1991": "Years since baseline (1991)",
                "decade": "Decade bin (1990s, 2000s, 2010s)",
                "is_1990s": "Binary flag: sample from 1990s",
                "is_2000s": "Binary flag: sample from 2000s",
                "is_2010s": "Binary flag: sample from 2010s"
            },
            "source": "Derived from year",
            "available_for_us": "Partial (year only)"
        },

        "environmental_demographic": {
            "name": "🌍 Environmental & Demographic",
            "description": "Socio-environmental context features (Europe-specific)",
            "features": {
                "PopulationDensity": "Population per square kilometer",
                "TerraMarineProtected_2016_2018": "% protected areas (2016-2018)",
                "TouristMean_1990_2020": "Average tourism metrics (1990-2020)",
                "VenueCount": "Number of venues/establishments",
                "netMigration_2011_2018": "Net migration rate (2011-2018)",
                "droughts_floods_temperature": "Climate extreme indicators",
                "literacyRate_2010_2018": "Literacy rate % (2010-2018)",
                "combustibleRenewables_2009_2014": "% renewable energy (2009-2014)",
                "gdp": "Gross Domestic Product"
            },
            "source": "European statistical databases",
            "available_for_us": False
        },

        "waste_management": {
            "name": "♻️ Waste Management",
            "description": "Waste composition and treatment metrics (Europe-specific)",
            "features": {
                "composition_food_organic_waste_percent": "% food/organic waste",
                "composition_glass_percent": "% glass waste",
                "composition_metal_percent": "% metal waste",
                "composition_other_percent": "% other waste",
                "composition_paper_cardboard_percent": "% paper/cardboard waste",
                "composition_plastic_percent": "% plastic waste",
                "composition_rubber_leather_percent": "% rubber/leather waste",
                "composition_wood_percent": "% wood waste",
                "composition_yard_garden_green_waste_percent": "% yard/garden waste",
                "waste_treatment_recycling_percent": "% waste recycled"
            },
            "source": "European waste management data",
            "available_for_us": False
        },

        "derived_water_quality": {
            "name": "🔬 Derived Water Quality",
            "description": "Calculated features from raw water quality parameters",
            "features": {
                "ph_deviation_from_7": "Absolute deviation from neutral pH (7.0)",
                "do_temp_ratio": "DO/temperature ratio (saturation proxy)",
                "conductance_low": "Binary flag: conductance < 200 µS/cm",
                "conductance_medium": "Binary flag: 200-800 µS/cm",
                "conductance_high": "Binary flag: conductance > 800 µS/cm",
                "pollution_stress": "Combined nitrate/DO pollution index",
                "temp_stress": "Temperature deviation from optimal (15°C)"
            },
            "source": "Calculated from WQI parameters",
            "available_for_us": True
        },

        "missing_indicators": {
            "name": "❓ Missing Value Indicators",
            "description": "Flags tracking parameter availability",
            "features": {
                "ph_missing": "Binary flag: pH measurement missing",
                "dissolved_oxygen_missing": "Binary flag: DO measurement missing",
                "temperature_missing": "Binary flag: temperature missing",
                "turbidity_missing": "Binary flag: turbidity missing",
                "nitrate_missing": "Binary flag: nitrate missing",
                "conductance_missing": "Binary flag: conductance missing",
                "n_params_available": "Count of available WQI parameters (0-6)"
            },
            "source": "Data availability tracking",
            "available_for_us": True
        },

        "water_body_type": {
            "name": "💧 Water Body Type",
            "description": "Type of water source (one-hot encoded)",
            "features": {
                "water_body_GW": "Groundwater source",
                "water_body_LW": "Lake water source",
                "water_body_RW": "River water source"
            },
            "source": "Sample metadata",
            "available_for_us": "Partial (US has some metadata)"
        },

        "geographic_country": {
            "name": "🗺️ Geographic (Country)",
            "description": "European country one-hot encoding (Europe-specific)",
            "features": {
                "country_Belgium": "Sample from Belgium",
                "country_Bulgaria": "Sample from Bulgaria",
                "country_Finland": "Sample from Finland",
                "country_France": "Sample from France",
                "country_Germany": "Sample from Germany",
                "country_Italy": "Sample from Italy",
                "country_Lithuania": "Sample from Lithuania",
                "country_Other": "Sample from other European country",
                "country_Serbia": "Sample from Serbia",
                "country_Spain": "Sample from Spain",
                "country_United Kingdom": "Sample from United Kingdom"
            },
            "source": "Sample location",
            "available_for_us": False
        },

        "derived_economic": {
            "name": "💰 Derived Economic",
            "description": "Calculated economic indicators (Europe-specific)",
            "features": {
                "gdp_per_capita_proxy": "GDP / population density (proxy for per capita)"
            },
            "source": "Calculated from GDP & population",
            "available_for_us": False
        }
    }


def count_features_by_availability() -> Dict[str, int]:
    """
    Count features by US data availability.

    Returns:
        Dictionary with counts for available, missing, and partial features
    """
    categories = get_feature_categories()

    available = 0
    missing = 0
    partial = 0

    for category_data in categories.values():
        n_features = len(category_data["features"])
        availability = category_data["available_for_us"]

        if availability is True:
            available += n_features
        elif availability is False:
            missing += n_features
        else:  # Partial availability
            partial += n_features

    return {
        "available": available,
        "missing": missing,
        "partial": partial,
        "total": available + missing + partial
    }


def get_european_only_features() -> List[str]:
    """
    Get list of features that are only available for European data.

    Returns:
        List of feature names that will be imputed for US predictions
    """
    categories = get_feature_categories()
    european_features = []

    for category_data in categories.values():
        if category_data["available_for_us"] is False:
            european_features.extend(category_data["features"].keys())

    return european_features


def get_us_available_features() -> List[str]:
    """
    Get list of features fully available for US data.

    Returns:
        List of feature names directly measurable from US water samples
    """
    categories = get_feature_categories()
    us_features = []

    for category_data in categories.values():
        if category_data["available_for_us"] is True:
            us_features.extend(category_data["features"].keys())

    return us_features
