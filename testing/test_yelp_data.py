import pandas as pd
import pytest
from yelp_data import (
    filter_businesses,
    clean_strings,
    clean_business_attributes,
    remove_nested_attributes,
    preprocess_features,
    get_original_attribute,
    aggregate_feature_importance
)

# ==============================================================================
# Test 1: Business Filtering
# Tests normal filtering behavior and handling of a missing category value
# ==============================================================================

def test_filter_businesses():
    test_data = pd.DataFrame({
        "categories": [
            "Restaurants, Italian",
            "Nail Salons, Beauty",
            "Restaurants, Mexican",
            None # Edge case: business has no category value
        ],
        "review_count": [
            600, # Meets both conditions
            900, # Has enough reviews, but not a restaurant
            200, # Restaurant, but not enough reviews
            700 # Enough reviews, but no category
        ]
    })

    result = filter_businesses(
        test_data,
        category="Restaurants",
        min_reviews=500
    )

# Only business that meets both requirements (Restaurant, minimum 500 reviews) should remain
    assert len(result) == 1
    assert result.iloc[0]["categories"] == "Restaurants, Italian"

# ==============================================================================
# Test 2: String Cleaning
# Tests expected string cleaning behavior and non-string edge case
# ==============================================================================

def test_clean_strings():
    assert clean_strings("u'free'") == "free"
    assert clean_strings("'average'") == "average"
    assert clean_strings("  quiet  ") == "quiet"

    # Edge case: non-string values should remain unchanged
    assert clean_strings(True) is True

# ==============================================================================
# Test 3: Business Attribute Cleaning
# Tests cleaning operations used on Yelp Attribute data
# ==============================================================================

def test_clean_business_attributes():
    test_attributes = pd.DataFrame({
        "WiFi": ["u'free'", "None"],
        "OutdoorSeating": ["True", "False"]
    })

    
    result = clean_business_attributes(test_attributes)

    # Unicode string should be cleaned
    assert result.iloc[0]["WiFi"] == "free"
    # Missing values should be converted to actual missing value
    assert pd.isna(result.iloc[1]["WiFi"])

    # Strings converted to actual booleans
    assert result.iloc[0]["OutdoorSeating"] == True
    assert result.iloc[1]["OutdoorSeating"] == False

# ==============================================================================
# Test 4: Nested Attribute Removal
# Tests normal column removal and behavior when a requested column is missing
# ==============================================================================  

def test_remove_nested_attributes():
    test_attributes = pd.DataFrame({
        "WiFi": ["free", "no"],
        "Alcohol": ["none", "full_bar"],
        "Ambience": ["casual", "romantic"],
        "BusinessParking": ["yes", "no"]
    })

    nested_attributes = [
        "Ambience",
        "BusinessParking"
    ]

    # Expected behavior: nested attributes should be removed
    result = remove_nested_attributes(
        test_attributes,
        nested_attributes
    )

    assert "Ambience" not in result.columns
    assert "BusinessParking" not in result.columns
    assert "WiFi" in result.columns
    assert "Alcohol" in result.columns

    # Test edge case: one requested column does not exist
    edge_result = remove_nested_attributes(
        test_attributes,
        ["Ambience", "DoesNotExist"]
    )

    assert "Ambience" not in edge_result.columns

# ==============================================================================
# Test 5: Feature Preprocessing
# Tests one-hot encoding, numeric feature handling, and preservation of rows
# ==============================================================================  

def test_preprocess_features():
    test_features = pd.DataFrame({
        "WiFi": ["free", "no", "free"],
        "OutdoorSeating": [True, False, True],
        "RestaurantsPriceRange2": ["1", "2", "3"]
    })

    x_encoded, preprocessor = preprocess_features(test_features)

     # Preprocessing should transform features without removing observations
    assert x_encoded.shape[0] == len(test_features)

    
    feature_names = preprocessor.get_feature_names_out()

    # Categorical columns should be one-hot encoded
    assert any("WiFi" in name for name in feature_names)
    assert any("OutdoorSeating" in name for name in feature_names)

    # Price range should remain a single numeric feature
    price_features = [
        name for name in feature_names
        if "RestaurantsPriceRange2" in name
    ]

    assert len(price_features) == 1

# ==============================================================================
# Test 6: Feature Importance
# Tests mapping encoded features back to their original Yelp attributes and aggregation of multiple encoded importance scores
# ==============================================================================

def test_feature_importance():
    attributes = [
        "WiFi",
        "OutdoorSeating",
        "Alcohol"
    ]

    # Test mapping encoded features back to original attributes
    assert get_original_attribute(
        "categorical__WiFi_free",
        attributes
    ) == "WiFi"

    assert get_original_attribute(
        "categorical__OutdoorSeating_True",
        attributes
    ) == "OutdoorSeating"

    # Test combining importance scores
    
    feature_importance = pd.DataFrame({
        "attribute": [
            "WiFi",
            "WiFi",
            "OutdoorSeating"
        ],
        "importance": [
            0.10,
            0.05,
            0.20
        ]
    })

    result = aggregate_feature_importance(feature_importance)

    # Encoded importance scores belonging to the same original attribute should be combined correctly
    assert result["WiFi"] == pytest.approx(0.15)
    assert result["OutdoorSeating"] == pytest.approx(0.20)

# ==============================================================================
# System / Integration Test
# Tests that the major parts of the analysis work together as one workflow
# Filtering -> cleaning -> preprocessing -> model training -> prediction -> evaluation
# ==============================================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

def test_end_to_end_workflow():
    # Create a small Yelp-like dataset
    test_data = pd.DataFrame({
        "categories": [
            "Restaurants, Italian",
            "Restaurants, Mexican",
            "Restaurants, Chinese",
            "Restaurants, American",
            "Restaurants, Thai",
            "Restaurants, Pizza",
            "Restaurants, Seafood",
            "Restaurants, Vietnamese"
        ],
        "review_count": [
            600, 700, 800, 900,
            550, 650, 750, 850
        ],
        "stars": [
            4.5, 3.5, 4.0, 3.0,
            4.5, 3.5, 4.0, 5.0
        ],
        "attributes": [
            {"WiFi": "u'free'", "OutdoorSeating": "True",
             "RestaurantsPriceRange2": "2"},
            {"WiFi": "u'no'", "OutdoorSeating": "False",
             "RestaurantsPriceRange2": "1"},
            {"WiFi": "u'free'", "OutdoorSeating": "True",
             "RestaurantsPriceRange2": "2"},
            {"WiFi": "u'no'", "OutdoorSeating": "False",
             "RestaurantsPriceRange2": "3"},
            {"WiFi": "u'free'", "OutdoorSeating": "True",
             "RestaurantsPriceRange2": "2"},
            {"WiFi": "u'no'", "OutdoorSeating": "False",
             "RestaurantsPriceRange2": "1"},
            {"WiFi": "u'free'", "OutdoorSeating": "True",
             "RestaurantsPriceRange2": "3"},
            {"WiFi": "u'free'", "OutdoorSeating": "False",
             "RestaurantsPriceRange2": "2"}
        ]
    })

    # Filter businesses
    filtered_data = filter_businesses(
        test_data,
        category="Restaurants",
        min_reviews=500
    )

    # Unpack and clean Yelp attributes
    attributes = filtered_data["attributes"].apply(pd.Series)
    clean_attributes = clean_business_attributes(attributes)

    # Remove nested attributes
    nested_attributes = [
        "BusinessParking",
        "Ambience",
        "GoodForMeal",
        "Music",
        "BestNights",
        "DietaryRestrictions"
    ]

    simple_attributes = remove_nested_attributes(
        clean_attributes,
        nested_attributes
    )

    # Prepare predictor and target variables
    x = simple_attributes
    y = filtered_data["stars"]

    # Preprocess features
    x_encoded, _ = preprocess_features(x)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded,
        y,
        test_size=0.25,
        random_state=42
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )

    model.fit(x_train, y_train)

    # Make predictions
    predictions = model.predict(x_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, predictions)
    # Filtering should retain all eight valid restaurant observations
    assert len(filtered_data) == 8
    # The model should produce exactly one prediction per test observation
    assert len(predictions) == len(y_test)
    # Predictions should contain valid finite numeric values
    assert np.isfinite(predictions).all()
    # MAE should be a valid non-negative error measurement
    assert mae >= 0
