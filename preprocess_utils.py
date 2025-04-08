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


