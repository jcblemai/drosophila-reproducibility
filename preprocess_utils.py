import pandas as pd
from sqlalchemy import create_engine, inspect

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
