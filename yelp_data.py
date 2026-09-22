"""
File: yelp_data.py
Author: Tien Le
Date: 2026-09-09

"""

# --- Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import polars as pl
import time


# ==============================================================================
# 1. Import the dataset
# ==============================================================================
data = pd.read_json('./data/yelp_academic_dataset_business.json', lines=True)


# ==============================================================================
# 2. Inspect the data
# ==============================================================================

print(data.shape) # Print shape: We get 150,346 rows and 14 columns
print(data.columns.tolist()) # View column names as list

# Explore the the categories column. From this, we can see that this dataset includes business categories that span from nail salons to restaurants. 
print(data["categories"].unique()) 
print(data["categories"].value_counts()) 


# ==============================================================================
# 3. Basic Filtering and Grouping
# ==============================================================================
# Function to filter categories and minimum reviews
def filter_businesses(data, category, min_reviews=500):
    category_mask = data["categories"].str.contains(
        category,
        na=False
    )

    review_count_mask = data["review_count"] >= min_reviews

    return data[category_mask & review_count_mask]

# Filter data to restaurants with at least 500 reviews
data = filter_businesses(data, "Restaurants")

print(data.head()) # Inspect the subset of data. 
print(data.shape) # The shape has now been reduced to (1263, 14)

# Exploring which state is most prevalent with restaurants with more than 500 reviews
print(data["state"].value_counts()) 
print(data.info())
print(data.describe())

data["attributes"].iloc[0] # Explore what is in the attributes field to explore what could be used in a ML algorithm

attributes = data["attributes"].apply(pd.Series) # Unpack attributes into a more readable format and explore
attributes.head()
attributes.columns.tolist() # Turn attributes into list

def clean_strings(value):
    if isinstance(value, str):
        value = value.strip()

        if value.startswith("u'") and value.endswith("'"):
            value = value[2:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]

    return value


def clean_business_attributes(attributes):
    clean_attributes = attributes.copy()

    # Replace variations of None with NaN
    clean_attributes = clean_attributes.replace(
        ["None", "None'", "u'None"],
        np.nan
    )

    # Clean formatting from strings
    clean_attributes = clean_attributes.map(clean_strings)

    # Convert True/False strings to booleans
    clean_attributes = clean_attributes.replace({
        "True": True,
        "False": False
    })

    return clean_attributes

# Call function 
clean_attributes = clean_business_attributes(attributes)

# Do a quick inspection of columns
print(clean_attributes["WiFi"].value_counts(dropna=False))
print(clean_attributes["NoiseLevel"].value_counts(dropna=False))
print(clean_attributes["Alcohol"].value_counts(dropna=False))


# To determine which attributes to focus on for the ML algorithm, we can calculate which attribute fields have the most values
attribute_percent = (clean_attributes.notna().sum() / len(clean_attributes) * 100).sort_values(ascending=False)
print(attribute_percent)

# Now within the attributes with higher percentage, we can look at the variability in each attribute. It's not useful to know that 99% of businesses accept credit cards, so we can look for attributes that are more split.
high_coverage = attribute_percent[attribute_percent >= 75].index

for column in high_coverage:
    print(f"\n{column}")
    print(clean_attributes[column].value_counts(dropna=False))

def remove_nested_attributes(attributes, nested_attributes):
    return attributes.drop(columns=nested_attributes, errors="ignore")

# For the purposes of this assignment, we can drop the fields with nested values
nested_attributes = [
    "BusinessParking",
    "Ambience",
    "GoodForMeal",
    "Music",
    "BestNights",
    "DietaryRestrictions"
]

simple_attributes = remove_nested_attributes(clean_attributes, nested_attributes)

# Join the original data with the simplified attributes data for analysis
restaurants_analysis = data[["stars", "review_count"]].join(simple_attributes)

simple_attributes_list = simple_attributes.columns.to_list()

for attribute in simple_attributes_list:
    print(f"\n . ݁₊ ⊹ . ݁ ⟡ ݁ {attribute} . ݁₊ ⊹ . ݁ ⟡  ")
    summary = restaurants_analysis.groupby(attribute,dropna=False)["stars"].agg(["count", "mean", "median"])
    print(summary)

# ==============================================================================
# 4. Explore a Machine Learning Algorithm
# ==============================================================================

# Use a machine learning algorithm to see if the algorithm can predict a star ratings based on a combination of attributes
x = restaurants_analysis[simple_attributes_list]
y = restaurants_analysis["stars"]

def preprocess_features(x):
    x = x.copy()

    categorical_columns = x.select_dtypes(
        include=["object", "str", "bool"]
    ).columns

    categorical_columns = [
        col for col in categorical_columns
        if col != "RestaurantsPriceRange2"
    ]

    x["RestaurantsPriceRange2"] = pd.to_numeric(
        x["RestaurantsPriceRange2"],
        errors="coerce"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    x_encoded = preprocessor.fit_transform(x)

    return x_encoded, preprocessor

x_encoded, preprocessor = preprocess_features(x)
x_encoded.shape # Check shape of encoded dataset

# Split the data into training and testing sets, 80% and 20% respectively. 
# Random state set for reproducibility 
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded,
    y,
    test_size=0.2,
    random_state=42
)

# Create a Random Forest Regression model using 200 decision trees
random_forest_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Train the model using training data
random_forest_model.fit(x_train, y_train)

# Use the trained model to predict ratings for the test data
rating_prediction = random_forest_model.predict(x_test)

print("Actual:", y_test.iloc[:5].values)
print("Predicted:", rating_prediction[:5])

mae = mean_absolute_error(y_test, rating_prediction)
print("Mean Absolute Error:", mae)

# Get the names and feature importance scores of the model features
feature_names = preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": random_forest_model.feature_importances_})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False)

print(feature_importance.head(10))

# Match each encoded feature back to the original restaurant attribute. Ex. categorical__OutdoorSeating_False and categorical__OutdoorSeating_True are separate right now, so combine them back to be Outdoor Seating overall)
def get_original_attribute(feature):
    for attribute in simple_attributes_list:
        if attribute in feature:
            return attribute

feature_importance["attribute"] = feature_importance["feature"].apply(get_original_attribute)

# Combine the scores of the encoded categories that fall under the same attribute
attribute_importance = (
    feature_importance
    .groupby("attribute")["importance"]
    .sum()
    .sort_values(ascending=False)
)

# Get the top 10 restaurant attributes
top_attributes = attribute_importance.head(10).index.tolist()

# Calculate average rating and restauarant count for each value within the top 10 attributes
results = []

for attribute in top_attributes:
    summary = (
        restaurants_analysis
        .groupby(attribute)["stars"]
        .agg(["mean", "count"])
        .reset_index()
    )
    
    summary["attribute"] = attribute
    summary = summary.rename(columns={attribute: "value"})
        
    results.append(summary)

# Create new dataframe
attribute_ratings = pd.concat(results, ignore_index=True)

print(attribute_importance.head(10))

# ==============================================================================
# Visualizations
# ==============================================================================

# Alcohol visualization
alcohol_data = attribute_ratings[attribute_ratings["attribute"] == "Alcohol"].sort_values("mean", ascending=False)
alcohol_data = alcohol_data.copy()

alcohol_data["value"] = alcohol_data["value"].replace({
    "full_bar": "Full Bar",
    "beer_and_wine": "Beer & Wine",
    "none": "No Alcohol"
})
plt.figure(figsize=(10, 4))

bar = plt.bar(
    alcohol_data["value"].astype(str),
    alcohol_data["mean"],
    width=0.4,
    color='pink'
    )
plt.bar_label(
    bar,
    fmt="%.2f",
    padding=3
)
plt.title("Average Yelp Rating for Restaurants with 500+ Reviews, by Alcohol Offering")
plt.xlabel("Alcohol Offering")
plt.ylabel("Average Yelp Rating", rotation=0,
           labelpad=60)
plt.ylim(3, 5)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "images/alcohol_ratings.png",
    dpi=300,
    bbox_inches="tight"
)

# Outdoor Seating Visualization
outdoor_seating_data = attribute_ratings[attribute_ratings["attribute"] == "OutdoorSeating"].sort_values("mean", ascending=False)
outdoor_seating_data = outdoor_seating_data.copy()

plt.figure(figsize=(10, 4))

bar = plt.bar(
    outdoor_seating_data["value"].astype(str),
    outdoor_seating_data["mean"],
    width=0.4,
    color='pink'
    )
plt.bar_label(
    bar,
    fmt="%.2f",
    padding=3
)
plt.title("Average Yelp Rating for Restaurants with 500+ Reviews, by Outdoor Seating Offering")
plt.xlabel("Outdoor Seating Offering")
plt.ylabel("Average Yelp Rating", rotation=0,
           labelpad=60)
plt.ylim(3, 5)
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "images/outdoor_ratings.png",
    dpi=300,
    bbox_inches="tight"
)


# plt.show()


# ==============================================================================
# Comparing pandas and polars
# ==============================================================================

# --- Test performance of filtering data ---
data_pandas = pd.read_json("./data/yelp_academic_dataset_business.json",lines=True)

start = time.perf_counter()
restaurants_pandas = data_pandas[data_pandas["categories"].str.contains("Restaurants", na=False)
                          &
                          data_pandas["review_count"] >= 500]

pandas_time_filter = time.perf_counter()-start
print(f"Pandas: {pandas_time_filter:.6f} seconds")


data_polars = pl.read_ndjson('./data/yelp_academic_dataset_business.json')
start = time.perf_counter()
restaurants_polars = data_polars.filter(
    pl.col("categories").str.contains("Restaurants")
    & 
    (pl.col("review_count") >= 500)
)
polars_time_filter = time.perf_counter()-start
print(f"Polars: {polars_time_filter:.6f} seconds")

# --- Test performance of grouping data and calculating summary statistics ---
start = time.perf_counter()
pandas_group_summary = (data_pandas.groupby("state")["stars"].agg(["mean", "count"]))
pandas_group_summary_time = time.perf_counter()-start

print(f"Pandas grouping and summary statistics: {pandas_group_summary_time:.6f} seconds")

start = time.perf_counter()
polars_group_summary = (data_polars.group_by("state").agg(pl.col("stars").mean().alias("mean"), pl.col("stars").count().alias("count")))
polars_group_time = time.perf_counter()-start

print(f"Polars grouping and summary statistics: {polars_group_time:.6f} seconds")


