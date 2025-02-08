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
        value = float(value)  
    except:
        return 0  

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

# Function to calculate final points
def calculate_points(row):
    try:
        volume = float(row["Volume"])
        normalized_competitor = float(row["All Competitor Score"])
        normalized_difficulty = float(row["Normalized Difficulty"])
        normalized_rank = float(row["Normalized Rank"])
        calculated_result = float(row["Calculated Result"])

        if normalized_difficulty == 0:  
            return 0

        points = (volume * normalized_competitor / normalized_difficulty) * normalized_rank * calculated_result
    except Exception:
        points = 0

    return points

##############################
# Part 2: Optimization Functions
##############################


def calculate_effective_points(keyword_list):
    """Calculate effective points per keyword and new keyword combinations based on total point."""
    def keyword_score(keyword, base_points):
        words = keyword.split()
        if len(words) == 1:
            return base_points  # Exact match (single word)
        return sum(base_points / (i + 1) for i in range(len(words) - 1))
    
    return [(kw, points, keyword_score(kw, points), keyword_score(kw, points), keyword_score(kw, points) * (1/3))
            for kw, points in keyword_list]

def sort_keywords_by_total_points(keyword_list):
    """Sort keywords by total calculated points instead of per character efficiency."""
    return sorted(keyword_list, key=lambda x: x[1], reverse=True)

def normalize_word(word):
    """Normalize words to handle singular/plural variations"""
    return word.rstrip('s')

def expand_keywords(keyword_list, max_length=29):
    """Generate potential keyword combinations based on existing keywords and calculate their adjusted points, ensuring max length constraint."""
    expanded_keywords = set(keyword_list)
    keyword_map = {kw: points for kw, points in keyword_list}

    for kw1, points1 in keyword_list:
        for kw2, points2 in keyword_list:
            if kw1 != kw2:
                words1 = kw1.split()
                words2 = kw2.split()

                # Combine words ensuring no duplicates
                combined = words1 + [w for w in words2 if w not in words1]

                # Ensure distinct words
                if len(set(combined)) != len(combined):
                    continue

                new_kw = " ".join(combined)

                # Check if new keyword fits character limit and is unique
                if new_kw not in keyword_map and new_kw not in expanded_keywords and len(new_kw) <= max_length:
                    # Handle common words for distance calculation
                    common_words = set(words1) & set(words2)
                    if common_words:
                        overlap_word = list(common_words)[0]  # Take the first common word
                        index1 = words1.index(overlap_word)
                        index2 = words2.index(overlap_word)

                        # Correct distance calculation: count words between occurrences
                        distance = abs((len(words1) - 1 - index1) + index2)
                        new_points = points1 + (points2 / (distance + 1))
                    else:
                        new_points = points1 + points2

                    # Final distinct word check
                    if len(set(new_kw.split())) == len(new_kw.split()):
                        expanded_keywords.add((new_kw, new_points))

    return list(expanded_keywords)

def construct_best_phrase(field_limit, keywords, multiplier, used_words, used_keywords):
    """Constructs the highest scoring phrase dynamically by combining keywords."""
    field = []
    total_points = 0
    remaining_chars = field_limit
    
    sorted_keywords = sort_keywords_by_total_points(keywords)
    while remaining_chars > 0 and sorted_keywords:
        best_keyword = sorted_keywords.pop(0)
        kw, base_points, f1_points, f2_points, f3_points = best_keyword
        words = kw.split()
        normalized_words = {normalize_word(word) for word in words}
        
        if kw not in used_keywords and not normalized_words.intersection(used_words):
            if remaining_chars - len(kw) >= 0:
                field.append(kw)
                total_points += base_points * field_limit * multiplier
                used_keywords.add(kw)
                used_words.update(normalized_words)
                remaining_chars -= len(kw) + 1  # +1 for space
    
    return field, total_points, used_keywords, field_limit - remaining_chars

def fill_field_with_word_breaking(field_limit, keywords, used_words, used_keywords, stop_words):
    """
    Fill Field 3 with word breaking, ensuring that adding a word (plus a comma if needed)
    does not exceed the field_limit (100 characters).
    """
    field = []
    total_points = 0
    remaining_chars = field_limit
    
    for kw, base_points, f1_points, f2_points, f3_points in keywords:
        if kw in used_keywords:
            continue  # Skip already used full keywords
        words = kw.split()
        for word in words:
            normalized_word = normalize_word(word)
            if normalized_word not in used_words and normalized_word not in stop_words:
                # Determine separator length: 1 character for a comma if field is not empty.
                sep_length = 1 if field else 0
                if remaining_chars - (len(word) + sep_length) >= 0:
                    field.append(word)
                    total_points += f3_points  # Full points if the word is used
                    used_words.add(normalized_word)
                    remaining_chars -= (len(word) + sep_length)
                else:
                    # Stop adding words if the next one doesn't fit.
                    break
    return field, total_points, used_keywords, field_limit - remaining_chars

def optimize_keyword_placement(keyword_list):
    """Optimize keyword placement across three fields for maximum points."""
    stop_words = {"the", "and", "for", "to", "of", "an", "a", "in", "on", "with", "by", "as", "at", "is","app","free"}
    expanded_keywords = expand_keywords(keyword_list, max_length=29)
    sorted_keywords = calculate_effective_points(expanded_keywords)
    used_words = set()
    used_keywords = set()
    
    # Construct best phrase dynamically for Field 1 (multiplier 1)
    field1, points1, used_kw1, length1 = construct_best_phrase(29, sorted_keywords, 1, used_words, used_keywords)
    
    # Construct best phrase dynamically for Field 2 (multiplier 1)
    field2, points2, used_kw2, length2 = construct_best_phrase(29, sorted_keywords, 1, used_words, used_keywords)
    
    # Fill Field 3 (multiplier 1/3, allows word breaking) with a 100-character limit
    field3, points3, used_kw3, length3 = fill_field_with_word_breaking(100, sorted_keywords, used_words, used_keywords, stop_words)
    points3 *= (1/3)
    
    # Join Field 3 keywords with a comma (no extra space)
    field3_str = ",".join(field3)
    # Ensure that the final string does not exceed 100 characters.
    if len(field3_str) > 100:
        field3_str = field3_str[:100]
    
    total_points = points1 + points2 + points3
    
    return {
        "Field 1": (" ".join(field1), points1, length1),
        "Field 2": (" ".join(field2), points2, length2),
        "Field 3": (field3_str, points3, len(field3_str)),
        "Total Points": total_points
    }


##############################
# Part 3: Streamlit Interface
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

    # Apply normalization to competitor columns
    for col in ["Competitor1", "Competitor2", "Competitor3", "Competitor4", "Competitor5"]:
        df[f"Normalized {col}"] = df[col].apply(normalize_competitor)

    # Create "All Competitor Score"
    df["All Competitor Score"] = df[
        ["Normalized Competitor1", "Normalized Competitor2", "Normalized Competitor3",
         "Normalized Competitor4", "Normalized Competitor5"]
    ].sum(axis=1) / 5

    # Ensure All Competitor Score is at least 1
    df["All Competitor Score"] = df["All Competitor Score"].apply(lambda x: 1 if x == 0 else x)

    # Apply normalization functions
    df["Normalized Difficulty"] = df["Difficulty"].apply(update_difficulty)
    df["Normalized Rank"] = df["Rank"].apply(update_rank)
    df["Calculated Result"] = df["Results"].apply(update_result)

    # Calculate "Final Points"
    df["Final Points"] = df.apply(calculate_points, axis=1)

    # Apply effective points calculation
    keyword_list = list(zip(df["Keyword"].tolist(), df["Final Points"].tolist()))
    effective_points_list = calculate_effective_points(keyword_list)

    # Convert effective points into a DataFrame
    effective_points_df = pd.DataFrame(effective_points_list, columns=[
        "Keyword", "Final Points", "Effective Points 1", "Effective Points 2", "Effective Points 3"
    ])

    # Merge with the main DataFrame
    df = df.merge(effective_points_df, on="Keyword", how="left")

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
