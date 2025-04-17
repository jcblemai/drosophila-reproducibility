from plot_info import assessment_columns
import pandas as pd

def create_author_metric(claim_df, variable, other_col: dict = {}):
    """
    Generates a DataFrame containing aggregated metrics for authors based on a specified variable.
    This function computes base metrics such as the count of major claims and unique articles 
    for each author (or other grouping variable). It also calculates the distribution of 
    assessment types and their proportions relative to the total number of major claims.
    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be analyzed.
        variable (str): The column name in the DataFrame to group by (e.g., author identifier).
        other_col (dict): Additional aggregation operations to include in the base metrics.
    Returns:
        pd.DataFrame: A DataFrame containing the aggregated metrics, including:
            - Base metrics: counts of major claims and unique articles.
            - Assessment type counts: counts of each assessment type.
            - Proportions: the proportion of each assessment type relative to the total major claims.

    """

    # Create base aggregation with name, counts, and article counts
    author_base = claim_df.groupby(variable).agg(**{
        "Major claims":('id', 'count'),
        "Articles":('article_id', 'nunique')
        }, **other_col
    )

    # Create a cross-tabulation of first_author_key and assessment_type_grouped
    assessment_counts = pd.crosstab(
        claim_df[variable], 
        claim_df['assessment_type_grouped']
    )

    # Make sure all assessment columns exist (some might be missing if no authors had that type)
    for col in assessment_columns:
        if col not in assessment_counts.columns:
            assessment_counts[col] = 0

    # Combine the base metrics with assessment counts
    author_metrics = pd.concat([author_base, assessment_counts], axis=1)

    for col in assessment_columns:
        author_metrics[f'{col} prop'] = author_metrics[col] / author_metrics['Major claims']

    author_metrics = author_metrics.reset_index()
    return author_metrics


def group_categorical_variable(df, variable: str):
    # Group by the categorical variable
    var_grouped = df.groupby(variable).agg({
        **{col: 'sum' for col in assessment_columns},
        'Major claims': 'sum',
        'Articles': 'sum',
    })
    
    # Calculate proportions
    for col in assessment_columns:
        var_grouped[f'{col}_prop'] = var_grouped[col] / var_grouped['Major claims']
    
    return var_grouped



def prepare_categorical_variable_data(df, author_metrics, variable, key_col, assessment_columns):
    """
    Prepare data for categorical variable comparison plots. By merging with the author metrics
    
    Parameters:
    -----------
    df : DataFrame
        The original claims dataframe with all columns
    author_metrics : DataFrame 
        The author metrics dataframe with first_author_key as index
    variable : str
        The categorical variable column name to analyze
    
    Returns:
    --------
    DataFrame
        Data prepared for plotting with the horizontal bar chart function
    """
    raise ValueError("This function is not correct.")
    # This was wrong because it overcounted claims when an author had different values for the same variable Utiliser pour les 2025-04-09
    
    # Get unique first author keys and their categorical variable values
    author_variable_mapping = df[[key_col, variable]].drop_duplicates().set_index(key_col)
    
    # Merge the author metrics with the categorical variable values
    combined_data = pd.merge(
        author_metrics, 
        author_variable_mapping, 
        left_on=key_col, 
        right_index=True,
        how='left'
    )
    
    # Group by the categorical variable
    var_grouped = combined_data.groupby(variable).agg({
        **{col: 'sum' for col in assessment_columns},
        'Major claims': 'sum',
        'Articles': 'sum',
    })
    
    # Calculate proportions
    for col in assessment_columns:
        var_grouped[f'{col}_prop'] = var_grouped[col] / var_grouped['Major claims']
    
    return var_grouped