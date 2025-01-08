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
