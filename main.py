import streamlit as st
import pandas as pd
import io
import itertools
from collections import defaultdict

##############################
# Part 1: Data Processing Code
##############################

# Function to perform word-level analysis on the keywords
def analyze_words(keywords, combined_text):
    # For each keyword, split into individual words and join them with a comma for display
    keywords_with_words = {phrase: ", ".join(phrase.split()) for phrase in keywords}
    # Normalize the combined text (handling patterns like "english,american,")
    combined_normalized = [word.strip().lower() for word in combined_text.replace(",", " ").split() if word.strip()]
    # Build the analysis: for each phrase, list its words and append those not found in the combined list
    results = {
        phrase: {
            "Split Words": keywords_with_words[phrase],
            "Status": ",".join([word for word in phrase.split() if word.lower() not in combined_normalized])
        }
        for phrase in keywords_with_words
    }
    # Convert the results dictionary into a DataFrame for display
    results_df = pd.DataFrame([
        {"Phrase": phrase,
         "Split Words": data["Split Words"],
         "Status": data["Status"]}
        for phrase, data in results.items()
    ])
    return results_df

# Define the stop words to be removed
stop_words = {"photos","images","the", "and", "for", "to", "of", "an", "a", "in", "on", "with", "by", "as", "at", "is", "app", "free"}

def remove_stop_words(text, stop_words):
    """Remove exact match stop words using regex and normalize spaces."""
    stop_words_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stop_words) + r')\b'
    
    # Remove stop words and clean spaces
    cleaned_text = re.sub(stop_words_pattern, '', text, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Ensure single spacing
    
    return cleaned_text


# Function to perform word-level analysis on the keywords
def analyze_words(keywords, combined_text):
    # For each keyword, split into individual words and join them with a comma for display
    keywords_with_words = {phrase: ", ".join(phrase.split()) for phrase in keywords}
    # Normalize the combined text (handling patterns like "english,american,")
    combined_normalized = [word.strip().lower() for word in combined_text.replace(",", " ").split() if word.strip()]
    # Build the analysis: for each phrase, list its words and append those not found in the combined list
    results = {
        phrase: {
            "Split Words": keywords_with_words[phrase],
            "Status": ",".join([word for word in phrase.split() if word.lower() not in combined_normalized])
        }
        for phrase in keywords_with_words
    }
    # Convert the results dictionary into a DataFrame for display
    results_df = pd.DataFrame([
        {"Phrase": phrase,
         "Split Words": data["Split Words"],
         "Status": data["Status"]}
        for phrase, data in results.items()
    ])
    return results_df

# Function to update/normalize the Difficulty column
def update_difficulty(diff):
    try:
        diff = float(diff)
    except:
        return None
    if 0 <= diff <= 5:
        return 0.5
    elif 6 <= diff <= 10:
        return 0.8
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

# Function to update/normalize the Rank column.
# If rank is empty or null, assign 250 before applying normalization rules.
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

# Function to update/normalize the Results column into a Calculated Result
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

def normalize_competitor(value):
    try:
        value = float(value)  # Convert to float to handle string inputs
    except ValueError:
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

# Function to calculate the Final Score based on the formula:
# (Volume / Normalized Difficulty) * Normalized Rank * Calculated Result
def calculate_final_score(row):
    try:
        volume = float(row["Volume"])
    except:
        volume = 0
    nd = row["Normalized Difficulty"]
    nr = row["Normalized Rank"]
    cr = row["Calculated Result"]
    ac = row["All Competitor Score"]
    try:
        final_score = (volume / nd) * nr * cr*ac
    except Exception:
        final_score = 0
    return final_score

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
    
    return [(kw, points, keyword_score(kw, points), keyword_score(kw, points), keyword_score(kw, points) * (1/1))
            for kw, points in keyword_list]

def sort_keywords_by_total_points(keyword_list):
    """Sort keywords by total calculated points instead of per character efficiency."""
    return sorted(keyword_list, key=lambda x: x[2], reverse=True)

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

def fill_field_with_word_breaking(field_limit, df_table, used_words, used_keywords, stop_words, field1, field2):
    """
    Optimized Field3 filling:
    - Uses df_table's first (Keyword) and last (Final Score) columns.
    - Prioritizes exact match keywords (full score).
    - Assigns partial match keywords half score.
    - Uses (score / character count) to ensure maximum efficiency.
    - Dynamically replaces inefficient keywords.
    - Stops filling at 96+ characters.
    """
    field = []
    total_points = 0
    remaining_chars = field_limit
    field1_words = set(field1.split())
    field2_words = set(field2.split())
    field3_words = set()  # Track used words in Field3 to prevent duplicates
    field_scores = {}  # Store scores per character for each added keyword
    kw_to_original = {}  # Maps added words back to their original keyword

    # Extract keywords and final scores from df_table
    keywords = list(zip(df_table.iloc[:, 0], df_table.iloc[:, -1]))  # First & Last column

    for kw_tuple in keywords:
        print(kw_tuple[0])
        kw = remove_stop_words(kw_tuple[0], stop_words)
        print(kw)
        f3_points = float(kw_tuple[1])  # Extract Final Score
        kw_words = set(kw.split())

        if kw in used_keywords:
            continue  # Skip already used keywords
            

        # **Exact Match Case** (Full Score)
        if kw in df_table.iloc[:, 0].values:  
            score = f3_points
            decay_factor = 1
            add_words = kw_words - field1_words - field2_words - field3_words  # New words to add

        # **Partial Match Case** (Half Score)  
        elif kw_words.intersection(field1_words) or kw_words.intersection(field2_words):
            score = f3_points * 0.5
            decay_factor = 1
            add_words = kw_words - field1_words - field2_words - field3_words  # New words to add

        # **Decay Case** (Words separated, apply distance-based decay)
        else:
            extra_words = len(kw_words - field1_words - field2_words)
            decay_factor = 1 / (extra_words + 1) if extra_words > 0 else 0.5
            score = f3_points * decay_factor
            add_words = kw_words - field1_words - field2_words - field3_words  # New words to add

        if not add_words:
            continue  # Skip if no new words to add

        # **Calculate keyword length based on added words**
        added_kw = ",".join(add_words)  # Join words with commas
        kw_length = len(added_kw.replace(",", ""))  # Remove commas to get true character count

        # **Calculate Score Per Character**
        score_per_char = score / max(1, kw_length)  # Avoid division by zero

        # **Check if it fits & optimize replacement**
        if remaining_chars - kw_length - 1 >= 0:  # Ensure space for a comma
            field.append(added_kw)
            field_scores[added_kw] = score_per_char  # Store score per character
            kw_to_original[added_kw] = kw  # Store original keyword
            total_points += score
            used_keywords.add(kw)
            field3_words.update(add_words)
            remaining_chars -= kw_length + 1  # Account for comma
        else:
            # **Try replacing a less efficient keyword**
            if field:
                worst_kw = min(field, key=lambda k: field_scores.get(k, 0), default=None)

                if worst_kw and score_per_char > field_scores.get(worst_kw, 0):
                    field.remove(worst_kw)
                    field.append(added_kw)

                    # Remove the original keyword from used_keywords safely
                    original_kw = kw_to_original.pop(worst_kw, None)
                    if original_kw:
                        used_keywords.discard(original_kw)

                    kw_to_original[added_kw] = kw  # Store new original keyword
                    used_keywords.add(kw)
                    field3_words.update(add_words)
                    field_scores[added_kw] = score_per_char  # Update score dictionary

        # **Stop if field reaches 96+ characters**
        if field_limit - remaining_chars >= 120:
            break

    # **Format Output**
    return field, total_points, used_keywords, field_limit - remaining_chars



def optimize_keyword_placement(keyword_list):
    """Optimize keyword placement across three fields for maximum points."""
    expanded_keywords = expand_keywords(keyword_list, max_length=29)
    sorted_keywords = calculate_effective_points(expanded_keywords)
    sorted_keywords=sort_keywords_by_total_points(sorted_keywords)
    
    used_words = set()
    used_keywords = set()
    
    # Construct best phrase dynamically for Field 1 (multiplier 1)
    field1, points1, used_kw1, length1 = construct_best_phrase(29, sorted_keywords, 1, used_words, used_keywords)
    print(field1)
    # Construct best phrase dynamically for Field 2 (multiplier 1)
    field2, points2, used_kw2, length2 = construct_best_phrase(29, sorted_keywords, 1, used_words, used_keywords)
    print(field2)
    # Fill Field 3 (multiplier 1/3, allows word breaking) with a 100-character limit
    field3, points3, used_kw3, lenth = fill_field_with_word_breaking(
    130,  # Field limit (100 characters for Field 3)
    df_table,  # The keyword list sorted by points
    used_words,  # Words already used
    used_keywords,  # Keywords already used
    stop_words,  # Common words to ignore
    " ".join(field1),  # Convert Field 1 list to a string
    " ".join(field2)   # Convert Field 2 list to a string
    )
    points3 *= (1/1)
    
    # Join Field 3 keywords with a comma (no extra space)
    field3_str = ",".join(field3)
    # Ensure that the final string does not exceed 100 characters.
    if len(field3_str) > 130:
        field3_str = field3_str[:130]
    
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
table_input = st.text_area("Paste your Excel table data", height=200)

if table_input:
    try:
        table_io = io.StringIO(table_input)
        df_table = pd.read_csv(table_io, sep="\t")
    except Exception as e:
        st.error(f"Error reading table data: {e}")
        st.stop()
    
    required_columns = ["Keyword", "Volume", "Difficulty", "Chance", "KEI", "Results", "Rank"]
    if not all(col in df_table.columns for col in required_columns):
        st.error(f"The pasted table must contain the following columns: {', '.join(required_columns)}")
        st.stop()
    else:
        # Normalize and calculate columns
        df_table["Normalized Difficulty"] = df_table["Difficulty"].apply(update_difficulty)
        df_table["Normalized Rank"] = df_table["Rank"].apply(update_rank)
        df_table["Calculated Result"] = df_table["Results"].apply(update_result)
        # Apply normalization to competitor columns and store in new columns
        for col in ["Competitor1", "Competitor2", "Competitor3", "Competitor4", "Competitor5"]:
            df_table[f"Normalized {col}"] = df_table[col].apply(normalize_competitor)
        # Create "All Competitor Score" as the sum of all normalized competitors divided by 5
        df_table["All Competitor Score"] = df_table[["Normalized Competitor1", "Normalized Competitor2", "Normalized Competitor3", 
                                 "Normalized Competitor4", "Normalized Competitor5"]].sum(axis=1) / 8
        df_table["Final Score"] = df_table.apply(calculate_final_score, axis=1)
        df_table = df_table.drop(columns=["Chance", "KEI"])
        df_table = df_table.sort_values(by="Final Score", ascending=False)
        
        # Build the keyword list for optimization from the Excel data:
        # Each tuple: (Keyword, Final Score)
        opt_keyword_list = list(zip(df_table["Keyword"].tolist(), df_table["Final Score"].tolist()))
        optimized_fields = optimize_keyword_placement(opt_keyword_list)
        
        # Extract all keywords (for word analysis) from the table
        excel_keywords = df_table["Keyword"].dropna().tolist()
        
        ##############################
        # Display Text Inputs and Optimized Results
        ##############################
        st.subheader("Enter Word Lists")
        
        # First text input and its optimized field (Field 1)
        first_field = st.text_input("Enter first text (max 30 characters)", max_chars=30)
        st.write("**Optimized Field 1:**", optimized_fields.get("Field 1")[0])
        
        # Second text input and its optimized field (Field 2)
        second_field = st.text_input("Enter second text (max 30 characters)", max_chars=30)
        st.write("**Optimized Field 2:**", optimized_fields.get("Field 2")[0])
        
        # Third text input and its optimized field (Field 3)
        third_field = st.text_input("Enter third text (comma or space-separated, max 100 characters)", max_chars=100)
        st.write("**Optimized Field 3:**", optimized_fields.get("Field 3")[0])
        
        # Combine the three fields for word analysis
        combined_text = f"{first_field} {second_field} {third_field}".strip()
        
        # Perform word analysis on the combined text using keywords from Excel
        analysis_df = analyze_words(excel_keywords, combined_text)
        st.write("### Word Analysis Results")
        st.dataframe(analysis_df, use_container_width=True)
        st.dataframe(df_table, use_container_width=True)
        st.download_button(
            label="Download Word Analysis CSV",
            data=analysis_df.to_csv(index=False, encoding="utf-8"),
            file_name="word_presence_analysis.csv",
            mime="text/csv"
        )
        
else:
    st.write("Please paste your table data to proceed.")
