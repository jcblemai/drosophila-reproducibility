import pandas as pd
from sqlalchemy import create_engine, inspect

def clean_df_from_database(df):
    # Create a copy of the DataFrame at the start
    df = df.copy()
    
    columns_to_remove = ['user_id', 'orcid_user_id', 
                    'created_at', 'updated_at', 'assertion_updated_at', 
                    'workspace_id', 'user_id', 'doi', 'organism_id', # 'pmid', 
                    'all_tags_json', 'obsolete', 'ext', 'badge_classes','pluralize_title',
                    'can_attach_file', 'refresh_side_panel', 'icon_classes', 'btn_classes']
    patterns_to_remove = ['validated', 'filename', 'obsolete_article']
    
    # Remove existing columns
    existing_cols = [col for col in columns_to_remove if col in df.columns]
    if existing_cols:
        df = df.drop(existing_cols, axis=1)
    
    # Remove pattern-matched columns
    pattern_cols = []
    for pattern in patterns_to_remove:
        pattern_cols.extend([c for c in df.columns if pattern in c])
    if pattern_cols:
        df = df.drop(pattern_cols, axis=1)
    
    return df

def build_author_key(df, author_name_col, key_col):
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Convert to lowercase and create the key column
        df[key_col] = df[author_name_col].str.lower()
        
        # Normalize unicode characters to ASCII
        df[key_col] = (df[key_col]
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8'))
        
        # Clean the keys using existing function
        df = clean_author_keys(df, key_col)
        
        return df

def clean_author_keys(df,  column):
    replacements = {
        'ando': 'ando i',
        'derre': 'derre i',
        'markus': 'markus r'
    }
    
    # Replace values only where they exactly match
    df.loc[df[column].isin(replacements.keys()), column] = \
        df.loc[df[column].isin(replacements.keys()), column].map(replacements)
    
    return df

def truncate_string(s, max_length=20):
    """Truncate string to max_length characters."""
    if isinstance(s, str) and len(s) > max_length:
        return s[:max_length] + '...'
    return s

def safe_strip(x):
    """
    Safely strip whitespace from strings in a DataFrame column and replace HTML entities.
    """
    if x.dtype == "object":
        return x.apply(lambda val: val.strip().replace('&amp;', '&') if isinstance(val, str) else val)
    return x

def load_all_tables(database='reprosci', host='localhost', user=None, password=None):
    """
    Load all non-empty tables from PostgreSQL database into DataFrames
    
    Parameters:
    -----------
    database : str
        Database name
    host : str
        Database host
    user : str, optional
        Database user
    password : str, optional
        Database password
        
    Returns:
    --------
    dict
        Dictionary of table_name: DataFrame pairs
    """
    
    # Create connection string
    if user and password:
        conn_string = f'postgresql://{user}:{password}@{host}/{database}'
    else:
        conn_string = f'postgresql://{host}/{database}'
    
    # Create engine
    engine = create_engine(conn_string)
    
    try:
        # Get inspector to get table info
        inspector = inspect(engine)
        
        # Get all table names
        tables = inspector.get_table_names()
        # sort tables by alphabetical order
        tables.sort()
        print(f"Found {len(tables)} tables: {', '.join(tables)}")

        
        # Dictionary to store DataFrames
        dfs = {}
        
        # Load each table
        for table in tables:
            # Check if table has rows
            row_count = pd.read_sql(f'SELECT COUNT(*) FROM {table}', engine).iloc[0,0]
            
            if True:#row_count > 0 and "gene" not in table: # table with gene are very big and not useful
                print(f"Loading {table} ({row_count} rows)")
                df = pd.read_sql_table(table, engine)
                dfs[table] = df
            else:
                print(f"Skipping empty table {table}")
        
        print(f"\nLoaded {len(dfs)} tables")
        
        return dfs
        
    finally:
        engine.dispose()

def deduplicate_by(df, col_name):
    """
    Deduplicate a dataframe based on a specific column, keeping the most common values 
    for other columns when duplicates exist.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to deduplicate
    col_name : str
        The column name to deduplicate by
        
    Returns:
    --------
    pandas.DataFrame
        Deduplicated dataframe with one row per unique value in col_name
    """
    from collections import Counter
    import pandas as pd
    import numpy as np
    
    # Create a list to store unique values and their most common attribute values
    unique_rows = []
    
    # Get unique values in the specified column
    unique_values = df[col_name].unique()
    
    # For each unique value
    for value in unique_values:
        # Get all rows with this value
        value_rows = df[df[col_name] == value]
        
        # Initialize a row for this unique value
        unique_row = {col_name: value}
        
        # For each column except the one we're deduplicating by
        for col in df.columns:
            if col == col_name:
                continue
                
            # Get the most common value
            values = value_rows[col].dropna().tolist()
            if len(values) == 0:
                unique_row[col] = np.nan
                continue
                
            # Use Counter to find the most common value
            value_counts = Counter(values)
            most_common_value, count = value_counts.most_common(1)[0]
            
            # Check if there are ties for most common value
            if sum(1 for v, c in value_counts.items() if c == count) > 1:
                print(f"Warning: Multiple most common values for {value:<15} in column {col:<10}. Choosing {most_common_value}")
                print(f"            among {values}.")
            
            unique_row[col] = most_common_value
        
        unique_rows.append(unique_row)
    
    # Create a new DataFrame from the unique values
    result_df = pd.DataFrame(unique_rows)
    
    # Reorder columns to match original DataFrame
    result_df = result_df[df.columns]
    
    return result_df


def add_shangai_ranking(df):
    df['shangai_ranking_2010'] = None
    # Now update the rankings based on the affiliations
    for idx, row in df.iterrows():
        aff = row["primary_affiliation"]
        if not pd.isna(aff):
            # Set the ranking based on affiliation
            if "Harvard" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 1
            elif "Stockholm" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 79
            elif "Tohoku" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 84
            elif "Umea" in aff or "Umeå" in aff or "Umeâ" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Maryland" in aff and "College Park" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 36
            elif "Worcester" in aff and "Massachusetts" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Washington" in aff and "Louis" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 30
            elif "Uppsala" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 66
            elif "Yonsei" in aff or "Notre Dame" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Petersburg" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 301
            elif "Stanford" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 3
            elif "La Jolla" in aff:
                df.at[idx, 'shangai_ranking_2010'] = None  # NA in original code
            elif "California" in aff:
                if "Davis" in aff:
                    df.at[idx, 'shangai_ranking_2010'] = 46
                elif "San Francisco" in aff:
                    df.at[idx, 'shangai_ranking_2010'] = 18
                elif "Berkeley" in aff:
                    df.at[idx, 'shangai_ranking_2010'] = 2
                elif "San Diego" in aff:
                    df.at[idx, 'shangai_ranking_2010'] = 14
                elif "Los Angeles" in aff:
                    df.at[idx, 'shangai_ranking_2010'] = 13
            elif "Cambridge" in aff and ("UK" in aff or "Kingdom" in aff):
                df.at[idx, 'shangai_ranking_2010'] = 5
            elif "Oxford" in aff and ("UK" in aff or "Kingdom" in aff):
                df.at[idx, 'shangai_ranking_2010'] = 10
            elif "University of Washington" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 16
            elif "Yale" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 11
            elif "University of Zürich" in aff or "Universität Zürich" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 51
            elif "Glasgow" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 151
            elif "Ohio University" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 401
            elif "Kiel" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 151
            elif "Texas" in aff and "A&M" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 95
            elif "University College London" in aff or ("UCL" in aff and "London" in aff):
                df.at[idx, 'shangai_ranking_2010'] = 21
            elif "Northwestern" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 29
            elif ("Zurich" in aff or "Zürich" in aff) and ("Federal Institute" in aff or "ETH" in aff):
                df.at[idx, 'shangai_ranking_2010'] = 23
            elif "Bonn" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 93
            elif "Houston" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Jiao Tong" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Pennsylvania" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 15
            elif "Rutgers" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 54
            elif "Oulu" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 301
            elif "Leuven" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Xiamen" in aff or "Pusan" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 401
            elif "City University" in aff and "New York" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Lund" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Brown University" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 65
            elif "University of Georgia" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Oregon State" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Canada" in aff and "Queen's" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "University of Alberta" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "University of Central Florida" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "China Agricultural University" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 401
            elif "Tel Aviv Uni" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Mount Sinai" in aff and "New York" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 151
            elif "Cornell University" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 12
            elif "McGill" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 61
            elif "Université Libre de Bruxelles" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "University of Birmingham" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 99
            elif "Massachusetts Institute of Technology" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 4
            elif "Kanazawa" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 301
            elif "Tufts" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "University of Wisconsin" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 17
            elif "Illinois" in aff and "Urbana" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 25
            elif "Illinois" in aff and "Chicago" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 151
            elif "Sichuan" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 301
            elif "University of Kentucky, Lexington" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "University of Calgary" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 151
            elif "University of Heidelberg" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 63
            elif "Aberdeen" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Kansas State University" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 301
            elif "Toulouse" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Universidad Autónoma" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif "Mayo" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 101
            elif "Pohang" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 301
            elif "University of Tokyo" in aff or "Tokyo University of Science" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 20
            elif "Korea Advanced Institute of Science and Technology" in aff:
                df.at[idx, 'shangai_ranking_2010'] = 201
            elif ("CNRS" in aff or "Centre National" in aff or "National Research Council, Montreal" in aff or
                "Tampere" in aff or "Ewha" in aff or "Novartis" in aff or
                "EPFL" in aff or "Ecole Polytechnique Fédérale de Lausanne" in aff or "bioGUNE" in aff or 
                "Institut de Biologie" in aff or "Justus Liebig University" in aff or
                "Université Louis Pasteur" in aff or "Aichi Cancer Center Research Institute" in aff or
                "Southern Methodist University" in aff or "Institute of Cancer Research" in aff or
                "Sloan-Kettering Institute" in aff or "Memorial Sloan-Kettering" in aff or
                "Hungarian" in aff or "Strasbourg" in aff or "Gutenberg" in aff or "DKFZ" in aff or
                "University of Aveiro" in aff or "Hebrew University, Rehovot" in aff or
                "Södertörns" in aff or "Whitehead" in aff or "Texas at El Paso" in aff or
                "Howard Hughes" in aff or "Vari, Greece" in aff or "Brigham and Women" in aff or
                "Borstel" in aff or "Osaka" in aff or "Children's Hospital" in aff or "Pasteur" in aff or
                "Tokyo Metropolitan University" in aff or "assachusetts General Hospital" in aff or "Sogang University" in aff or 
                "University of Modena and Reggio Emilia" in aff or "University of Missouri-Kansas" in aff or
                ("CEA" in aff and "Grenoble" in aff) or "Commissariat à l'Energie Atomique" in aff or "Obihiro" in aff or "UMR INRA-UM2" in aff or
                "National Institutes of Health" in aff or "Chinese Academy of Sciences" in aff or "Bohemia" in aff):
                df.at[idx, 'shangai_ranking_2010'] = pd.NA
    return df

# # 2010 ranking: from my manual search, updated by claude above ^
# for aff, cnt in major_claims_df["primary_affiliation"].value_counts().items():
#     if "Harvard" in aff:# rank: 1
#     elif "Stockholm" in aff:# rank: 79
#     elif "Tohoku" in aff:# rank: 84
#     elif "Umea" in aff or "Umeå" in aff or "Umeâ" in aff:# rank: 201
#     elif "Maryland" in aff and "College Park" in aff:# rank: 36
#     elif "Worcester" in aff and "Massachusetts" in aff:# rank: 101
#     elif "Washington" in aff and "Louis" in aff:# rank: 30
#     elif "Uppsala" in aff:# rank: 66
#     elif "Yonsei" in aff or "Notre Dame" in aff:# rank: 201
#     elif "Petersburg" in aff:# rank: 301
#     elif "Stanford" in aff:# rank: 3
#     elif "La Jolla" in aff:
#             pass
#             # rank: NA
#     elif "California" in aff:
#         if "Davis" in aff:
#             pass
#             # rank: 46
#         elif "San Francisco" in aff:
#             pass
#             # rank: 18
#         elif "Berkeley" in aff:
#             pass
#             # rank: 2
#         elif "San Diego" in aff:
#             pass
#             # rank: 14
#         elif "Los Angeles" in aff:
#             pass
#             # rank: 13
#         
#         else:
#             print(f" {cnt:3}: {aff}")
#     elif "Cambridge" in aff and ("UK" in aff or "Kingdom" in aff):# rank: 5
#     elif "Oxford" in aff and ("UK" in aff or "Kingdom" in aff):# rank: 10
#     elif "University of Washington" in aff:# rank: 16
#     elif "Yale" in aff:# rank: 11
#     elif "University of Zürich" in aff or "Universität Zürich" in aff:# rank: 51
#     elif "Glasgow" in aff:# rank: 151
#     elif "Ohio University" in aff:# rank: 401
#     elif "Kiel" in aff:# rank: 151
#     elif "Texas" in aff and "A&M" in aff:# rank: 95
#     elif "University College London" in aff or ("UCL" in aff and "London" in aff):# rank: 21
#     elif "Northwestern" in aff:# rank: 29
#     elif ("Zurich" in aff or "Zürich") and ("Federal Institute" in aff or "ETH" in aff):# rank: 23
#     elif "Bonn" in aff:# rank: 93
#     elif "Houston" in aff:# rank: 201
#     elif "Jiao Tong" in aff:# rank: 201
#     elif "Pennsylvania" in aff:# rank: 15
#     elif "Rutgers" in aff:# rank: 54
#     elif "Oulu" in aff:# rank: 301
#     elif "Leuven" in aff:# rank: 101
#     elif "Xiamen" in aff or "Pusan" in aff:# rank: 401
#     elif "City University" in aff and "New York" in aff:# rank: 201
#     elif "Lund" in aff:# rank: 101
#     elif "Brown University" in aff:# rank: 65
#     elif "University of Georgia" in aff:# rank: 101
#     elif "Oregon State"  in aff:# rank: 101
#     elif "Canada" in aff and "Queen's" in aff:# rank: 201
#     elif "University of Alberta" in aff:# rank: 101
#     elif " University of Central Florida" in aff:# rank: 201
#     elif "China Agricultural University" in aff:# rank: 401
#     elif "Tel Aviv Uni" in aff:# rank: 101
#     elif "Mount Sinai" in aff and "New York" in aff:# rank: 151
#     elif "Cornell University" in aff:# rank: 12
#     elif "McGill" in aff:# rank: 61
#     elif "Université Libre de Bruxelles" in aff:# rank: 101
#     elif "University of Birmingham" in aff:# rank: 99
#     elif "Massachusetts Institute of Technology" in aff:# rank: 4
#     elif "Kanazawa" in aff:# rank: 301
#     elif "Tufts" in aff:# rank: 101
#     elif "University of Wisconsin" in aff:# rank: 17
#     elif "Illinois" in aff and "Urbana" in aff:# rank: 25
#     elif "Illinois" in aff and "Chicago" in aff:# rank: 151
#     elif "Sichuan" in aff:# rank: 301
#     elif "University of Kentucky, Lexington" in aff:# rank: 201
#     elif "University of Calgary" in aff:# rank: 151
#     elif "University of Heidelberg" in aff:# rank: 63
#     elif "Aberdeen" in aff:# rank: 201
#     elif "Kansas State University" in aff:# rank: 301
#     elif "Toulouse" in aff:# rank: 201
#     elif "Universidad Autónoma" in aff:# rank: 201
#     elif "Mayo" in aff:# rank: 101
#     elif "Pohang" in aff:# rank: 301
#     elif "University of Tokyo" in aff or "Tokyo University of Science" in aff:# rank: 20
#     elif "Korea Advanced Institute of Science and Technology" in aff:# rank: 201
#     elif ("CNRS" in aff or "Centre National" in aff or "National Research Council, Montreal" in aff or
#         "Tampere" in aff or "Ewha" in aff or "Novartis" in aff or
#         "EPFL" in aff or "Ecole Polytechnique Fédérale de Lausanne" in aff or "bioGUNE" in aff or 
#         "Institut de Biologie" in aff or "Justus Liebig University" in aff or
#         " Université Louis Pasteur" in aff or " Aichi Cancer Center Research Institute" in aff or
#         "Southern Methodist University" in aff or "Institute of Cancer Research" in aff or
#         "Sloan-Kettering Institute" in aff or "Memorial Sloan-Kettering" in aff or
#         "Hungarian" in aff or "Strasbourg" in aff or "Gutenberg" in aff or "DKFZ" in aff or
#         "University of Aveiro" in aff or "Hebrew University, Rehovot" in aff or
#         "Södertörns" in aff or "Whitehead" in aff or "Texas at El Paso" in aff or
#         "Howard Hughes" in aff or "Vari, Greece" in aff or "Brigham and Women" in aff or
#         "Borstel" in aff or "Osaka" in aff or "Children's Hospital" in aff or "Pasteur" in aff or
#         "Tokyo Metropolitan University" in aff or "assachusetts General Hospital" in aff or "Sogang University" in aff or 
#         "University of Modena and Reggio Emilia" in aff or "University of Missouri-Kansas" in aff or
#         ("CEA" in aff and "Grenoble" in aff) or "Commissariat à l'Energie Atomique" in aff or "Obihiro" in aff or "UMR INRA-UM2" in aff or
#         "National Institutes of Health" in aff or "Chinese Academy of Sciences" in aff or "Bohemia" in aff):
#         # rank: NA
#         pass
# 
#     else:
#         print(f" {cnt:3}: {aff}")