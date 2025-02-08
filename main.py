# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:39:43 2025

@author: volka
"""

import streamlit as st
import pandas as pd
import io

st.title("Keyword Analysis with Competitor Normalization & Score Calculation")

##############################
# Part 1: Data Processing Code
##############################

# Function to update/normalize the Difficulty column
def update_difficulty(diff):
    try:
        diff = float(diff)
    except:
        return None
    if 0 <= diff <= 5:
        return 0.3
    elif 6 <= diff <= 10:
        return 0.5
    elif 11 <= diff <= 20:
        return 1
    elif 21 <= diff <= 30:
        return 2
    elif 31 <= diff <= 40:
        return 4
    elif 40 < diff <= 70:
        return 8
    elif 71 < diff <= 100:
        return 12
    else:
        return 1.0

# Function to update/normalize the Rank column
def update_rank(rank):
    try:
        rank = float(rank)
    except:
        rank = 250.0
    if 1 <= rank <= 10:
        return 5
    elif 11 <= rank <= 30:
        return 4
    elif 31 <= rank <= 50:
        return 4
    elif 51 <= rank <= 249:
        return 2
    else:
        return 1

# Function to update/normalize the Results column
def update_result(res):
    try:
        res = float(res)
    except:
        return 1
    if 1 <= res <= 20:
        return 3
    elif 21 <= res <= 50:
        return 2
    elif 51 <= res <= 200:
        return 1.5
    else:
        return 1

# Function to normalize competitor values
def normalize_competitor(value):
    try:
        value = float(value)  # Convert to float to handle string inputs
    except:
        return 0  # Return 0 if conversion fails (e.g., if value is non-numeric)

    if 1 <= value <= 10:
        return 5
    elif 11 <= value <= 20:
        return 4.5
    elif 21 <= value <= 30:
        return 4.2
    elif 31 <= value <= 60:
        return 4
    elif 61 <= value <= 100:
        return 3
    else:
        return 0

# Function to calculate final points based on the given formula
def calculate_points(row):
    try:
        volume = float(row["Volume"])
        normalized_competitor = float(row["All Competitor Score"])
        normalized_difficulty = float(row["Normalized Difficulty"])
        normalized_rank = float(row["Normalized Rank"])
        calculated_result = float(row["Calculated Result"])

        if normalized_difficulty == 0:  # Avoid division by zero
            return 0

        points = (volume * normalized_competitor / normalized_difficulty) * normalized_rank * calculated_result
    except Exception:
        points = 0

    return points


##############################
# Part 2: Streamlit Interface
##############################

# Text area for pasting table data
table_input = st.text_area("Paste your Excel table data (tab-separated)", height=200)

if table_input:
    try:
        table_io = io.StringIO(table_input)
        df = pd.read_csv(table_io, sep="\t")
    except Exception as e:
        st.error(f"Error reading table data: {e}")
        st.stop()

    required_columns = [
        "Keyword", "Volume", "Difficulty", "Results", "Rank", 
        "Competitor1", "Competitor2", "Competitor3", "Competitor4", "Competitor5"
    ]
    
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing columns in the data. Required: {', '.join(required_columns)}")
        st.stop()

    # Apply normalization to competitor columns and store in new columns
    for col in ["Competitor1", "Competitor2", "Competitor3", "Competitor4", "Competitor5"]:
        df[f"Normalized {col}"] = df[col].apply(normalize_competitor)

    # Create "All Competitor Score" as the sum of all normalized competitors divided by 5
    df["All Competitor Score"] = df[
        ["Normalized Competitor1", "Normalized Competitor2", "Normalized Competitor3",
         "Normalized Competitor4", "Normalized Competitor5"]
    ].sum(axis=1) / 5

    # Ensure All Competitor Score is at least 1 (avoid zero values)
    df["All Competitor Score"] = df["All Competitor Score"].apply(lambda x: 1 if x == 0 else x)

    # Apply normalization functions to the DataFrame
    df["Normalized Difficulty"] = df["Difficulty"].apply(update_difficulty)
    df["Normalized Rank"] = df["Rank"].apply(update_rank)
    df["Calculated Result"] = df["Results"].apply(update_result)

    # Apply the final points calculation
    df["Final Points"] = df.apply(calculate_points, axis=1)

    # Display the updated DataFrame
    st.write("### Processed Data with Normalization and Final Score Calculation")
    st.dataframe(df, use_container_width=True)

    # Download button for the processed data
    st.download_button(
        label="Download Processed Data as CSV",
        data=df.to_csv(index=False, encoding="utf-8"),
        file_name="processed_keyword_analysis.csv",
        mime="text/csv"
    )
