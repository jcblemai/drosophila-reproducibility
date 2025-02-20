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
# read an xlx file in pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the data
author_info = pd.read_excel('input_data/2025-02-14_last_xlsx/1_Triage_Last author.xlsx', sheet_name='Tri sans les doublons')
author_info.to_csv('input_data/2025-02-14_last_xlsx/1_Triage_Last author.csv', index=False)
first_authors_claims = pd.read_excel('input_data/2025-02-14_last_xlsx/stats_author.xlsx', sheet_name='First')
leading_authors_claims = pd.read_excel('input_data/2025-02-14_last_xlsx/stats_author.xlsx', sheet_name='Leading')
leading_authors_claims["Authorship"]= "Leading"
first_authors_claims["Authorship"]= "First"


authors_claims = pd.concat([leading_authors_claims, first_authors_claims])
authors_claims['Sex'] = authors_claims['Sex'].map({1: 'Male', 0: 'Female'})
authors_claims = authors_claims.drop(columns=[col for col in authors_claims.columns if '%' in col])

authors_claims.to_csv('input_data/2025-02-14_last_xlsx/stats_author.csv', index=False)

# %%
authors_claims.rename(columns={'Conituinity': 'Continuity'}, inplace=True)
authors_claims.rename(columns={'Conituinity': 'Continuity'}, inplace=True)
authors_claims['Historical lab'] = authors_claims['Historical lab'].astype('boolean')
authors_claims['Continuity'] = authors_claims['Continuity'].astype('boolean')
authors_claims

# %%
# Drop all columns with 'Unnamed' in the name
author_info = author_info.drop(columns=[col for col in author_info.columns if 'Unnamed' in col]).drop(columns=['Source'])


# %% [markdown]
# ## Let's go for last authors first

# %%
unique_leading_author = authors_claims[authors_claims['Authorship'] == 'Leading']

# %%
author_info

# %%


def aggregate_author_info(df):
    """
    Groups author information by last author and aggregates other columns 
    by taking the most common value.
    
    Args:
        df: pandas DataFrame with author information
    Returns:
        DataFrame grouped by last author with most common values for other columns
    """
    
    # Define aggregation function to get most common value
    def most_common(series):
        # Return first value if all are null/nan
        if series.isna().all():
            return pd.NA
        # Get most common non-null value
        return series.mode().iloc[0]
    
    # Group by last author and aggregate other columns
    grouped = df.groupby('last author').agg({
        'first author': 'first',  # Take first value for first author
        'Sex': most_common,
        'PhD Post-doc': most_common,
        'Become a Pi': most_common,
        'current job': most_common,
        'MD': most_common,
        'Affiliation': 'first',
        'Country': most_common,
        'Ivy league': most_common
    }).reset_index()
    
    # Clean up column names
    grouped.columns = [col.strip() for col in grouped.columns]
    
    return grouped

# Example usage:
grouped_authors_info = aggregate_author_info(author_info)
grouped_authors_info

# %%
author_info

# %%

leading_authors_df = pd.merge(unique_leading_author, grouped_authors_info.drop(columns=['Sex']), 
left_on='Name', 
right_on='last author', how='left').drop(columns=['last author', 'first author'])

# %%
leading_authors_df

# %% [markdown]
# leading_authors_df

# %%
leading_authors_df.to_csv('test.csv', index=False)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Read the data
df = pd.read_csv('test.csv')

# Calculate percentage of challenged claims for each author
df['challenged_rate'] = df['Challenged'] / df['Major claims'] * 100

# Function for statistical testing
def compare_groups(data, column, rate_column='challenged_rate'):
    groups = data[column].unique()
    if len(groups) == 2:  # For binary variables like Sex
        group1 = data[data[column] == groups[0]][rate_column]
        group2 = data[data[column] == groups[1]][rate_column]
        stat, pval = stats.mannwhitneyu(group1, group2)
        return {
            'groups': groups,
            'medians': [group1.median(), group2.median()],
            'p_value': pval
        }
    else:  # For variables with more than 2 categories
        stat, pval = stats.kruskal(*[group[rate_column].values 
                                    for name, group in data.groupby(column)])
        return {
            'groups': groups,
            'medians': [group[rate_column].median() 
                       for name, group in data.groupby(column)],
            'p_value': pval
        }

# Create subplots for our analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.3)

# 1. Sex Analysis
sns.boxplot(data=df, x='Sex', y='challenged_rate', ax=axes[0,0])
sex_stats = compare_groups(df, 'Sex')
axes[0,0].set_title(f'Challenged Claims Rate by Sex\np={sex_stats["p_value"]:.3f}')

# 2. Historical Lab Analysis
sns.boxplot(data=df, x='Historical lab', y='challenged_rate', ax=axes[0,1])
lab_stats = compare_groups(df, 'Historical lab')
axes[0,1].set_title(f'Challenged Claims Rate by Historical Lab\np={lab_stats["p_value"]:.3f}')

# 3. Country Analysis
sns.boxplot(data=df, x='Country', y='challenged_rate', ax=axes[1,0])
country_stats = compare_groups(df, 'Country')
axes[1,0].set_title(f'Challenged Claims Rate by Country\np={country_stats["p_value"]:.3f}')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Ivy League vs Non-Ivy League
sns.boxplot(data=df, x='Ivy league', y='challenged_rate', ax=axes[1,1])
ivy_stats = compare_groups(df, 'Ivy league')
axes[1,1].set_title(f'Challenged Claims Rate by Ivy League Status\np={ivy_stats["p_value"]:.3f}')

# Add overall title
plt.suptitle('Factors Affecting Rate of Challenged Claims', fontsize=16, y=1.02)

# Print summary statistics
print("\nSummary Statistics:")
for factor in ['Sex', 'Historical lab', 'Country', 'Ivy league']:
    stats_result = compare_groups(df, factor)
    print(f"\n{factor}:")
    for group, median in zip(stats_result['groups'], stats_result['medians']):
        print(f"{group}: Median challenged rate = {median:.2f}%")
    print(f"p-value = {stats_result['p_value']:.3f}")

# Additional analysis for continuous relationships
if 'Continuity' in df.columns:
    correlation = stats.spearmanr(df['Continuity'], df['challenged_rate'])
    print("\nContinuity correlation:")
    print(f"Spearman correlation = {correlation.correlation:.3f}")
    print(f"p-value = {correlation.pvalue:.3f}")

# Save the figure
plt.savefig('challenged_claims_analysis.png', bbox_inches='tight', dpi=300)

# Create a summary table
summary_df = df.groupby(['Sex', 'Historical lab', 'Country']).agg({
    'challenged_rate': ['mean', 'median', 'std', 'count']
}).round(2)

print("\nDetailed Summary Table:")
print(summary_df)
