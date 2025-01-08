# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import utils

from_database = True

if from_database:
    # Load tables from database
    dfs = utils.load_all_tables()
    print("\nSummary of loaded tables:")
    for table_name, df in dfs.items():
        print(f"\n{table_name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {', '.join(df.columns[:5])}...")
else:
    dfs = utils.load_dfs(method='pickle')


# %%
utils.save_dfs(dfs, method='pickle')

# %%
print("\n".join(dfs.keys()))

# %%
import pandas as pd
from sqlalchemy import create_engine, inspect
import re

def load_table_with_relations(table_name, database='reprosci', host='localhost', user=None, password=None):
    """
    Load a table and automatically merge in related data based on foreign keys
    
    Parameters:
    -----------
    table_name : str
        Name of the main table to load
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
    pd.DataFrame
        DataFrame with all related data merged in
    """
    
    # Create connection string
    if user and password:
        conn_string = f'postgresql://{user}:{password}@{host}/{database}'
    else:
        conn_string = f'postgresql://{host}/{database}'
    
    # Create engine
    engine = create_engine(conn_string)
    
    try:
        # Load the main table
        df = pd.read_sql_table(table_name, engine)
        print(f"Loaded main table '{table_name}' with {len(df)} rows")
        
        # Get column names
        columns = df.columns.tolist()
        
        # Find potential foreign key columns (ending in _id)
        fk_columns = [col for col in columns if col.endswith('_id')]
        
        # Process each foreign key
        for fk_col in fk_columns:
            # Get the referenced table name (remove _id suffix)
            ref_table = fk_col[:-3] + 's'  # e.g., journal_id -> journals
            
            try:
                # Load the referenced table
                ref_df = pd.read_sql_table(ref_table, engine)
                print(f"Processing relation: {table_name}.{fk_col} -> {ref_table}")
                
                # Merge with main dataframe
                df = df.merge(
                    ref_df,
                    how='left',
                    left_on=fk_col,
                    right_on='id',
                    suffixes=('', f'_{ref_table}')
                )
                
                print(f"Merged {ref_table} data")
                
            except Exception as e:
                print(f"Could not process relation for {fk_col}: {str(e)}")
        
        return df
        
    finally:
        engine.dispose()

# Example usage
if __name__ == "__main__":
    # Load articles with all relations
    articles_df = load_table_with_relations('articles')
    
    print("\nFinal DataFrame columns:")
    print(articles_df.columns.tolist())
    
    # Example showing a row with joined data
    sample = articles_df.iloc[0]
    print("\nSample row with joined data:")
    print(f"Article title: {sample.get('title')}")
    print(f"Journal name: {sample.get('name')}")  # from journals table

# %%
articles_df

# %%
dfs["journals"]

# %%
dfs["articles"]

# %%
dfs["assertions"]

