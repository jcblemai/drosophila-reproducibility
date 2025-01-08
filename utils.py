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


        import pickle
import os

def save_dfs(dfs, directory='saved_dfs', method='parquet'):
    """
    Save dictionary of DataFrames to disk
    
    Parameters:
    -----------
    dfs : dict
        Dictionary of DataFrames
    directory : str
        Directory to save files in
    method : str
        'pickle' or 'parquet'
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    if method == 'pickle':
        # Save all DataFrames in one pickle file
        with open(f'{directory}/all_tables.pkl', 'wb') as f:
            pickle.dump(dfs, f)
        print(f"Saved all tables to {directory}/all_tables.pkl")
            
    elif method == 'parquet':
        # Save each DataFrame as a separate parquet file
        for table_name, df in dfs.items():
            file_path = f'{directory}/{table_name}.parquet'
            df.to_parquet(file_path)
        
        # Save table names
        with open(f'{directory}/table_names.txt', 'w') as f:
            f.write('\n'.join(dfs.keys()))
        
        print(f"Saved {len(dfs)} tables to {directory}/")

def load_dfs(directory='saved_dfs', method='parquet'):
    """
    Load dictionary of DataFrames from disk
    
    Parameters:
    -----------
    directory : str
        Directory containing saved files
    method : str
        'pickle' or 'parquet'
        
    Returns:
    --------
    dict
        Dictionary of DataFrames
    """
    if method == 'pickle':
        # Load all DataFrames from pickle file
        with open(f'{directory}/all_tables.pkl', 'rb') as f:
            dfs = pickle.load(f)
        print(f"Loaded {len(dfs)} tables from {directory}/all_tables.pkl")
        
    elif method == 'parquet':
        # Get table names
        with open(f'{directory}/table_names.txt', 'r') as f:
            table_names = f.read().splitlines()
        
        # Load each DataFrame
        dfs = {}
        for table_name in table_names:
            file_path = f'{directory}/{table_name}.parquet'
            dfs[table_name] = pd.read_parquet(file_path)
        
        print(f"Loaded {len(dfs)} tables from {directory}/")
    
    return dfs

