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

# Read Author info, which contains all the pairs
paper_auth_pairs = pd.read_excel('input_data/2025-02-14_last_xlsx/1_Triage_Last author.xlsx', sheet_name='Tri sans les doublons')
# Drop all columns with 'Unnamed' in the name
paper_auth_pairs = paper_auth_pairs.drop(columns=[col for col in paper_auth_pairs.columns if 'Unnamed' in col]).drop(columns=['Source'])
paper_auth_pairs.to_csv('input_data/2025-02-14_last_xlsx/1_Triage_Last author.csv', index=False)

first_authors_claims = pd.read_excel('input_data/2025-02-14_last_xlsx/stats_author.xlsx', sheet_name='First')
leading_authors_claims = pd.read_excel('input_data/2025-02-14_last_xlsx/stats_author.xlsx', sheet_name='Leading')
leading_authors_claims["Authorship"]= "Leading"
first_authors_claims["Authorship"]= "First"


authors_claims = pd.concat([leading_authors_claims, first_authors_claims])
authors_claims['Sex'] = authors_claims['Sex'].map({1: 'Male', 0: 'Female'})
authors_claims = authors_claims.drop(columns=[col for col in authors_claims.columns if '%' in col])
authors_claims.rename(columns={'Conituinity': 'Continuity'}, inplace=True)
authors_claims['Historical lab'] = authors_claims['Historical lab'].astype('boolean')
authors_claims['Continuity'] = authors_claims['Continuity'].astype('boolean')
authors_claims.to_csv('input_data/2025-02-14_last_xlsx/stats_author.csv', index=False)


# %% [markdown]
# It seems that
# - sex -> FH
# - PhD Post-doc -> FH
# - Become a Pi -> FH
# - current job -> FH
# - MD -> **???**
# - Affiliation -> Both
# - Country -> Both
# - Ivy League -> Both

# %%
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
                print(f"Warning: Multiple most common values for {value} in column {col}. Choosing {most_common_value}")
            
            unique_row[col] = most_common_value
        
        unique_rows.append(unique_row)
    
    # Create a new DataFrame from the unique values
    result_df = pd.DataFrame(unique_rows)
    
    # Reorder columns to match original DataFrame
    result_df = result_df[df.columns]
    
    return result_df

# %%
paper_auth_pairs_LH = paper_auth_pairs[["last author", "Affiliation", "Country", "Ivy league"]]
paper_auth_pairs_LH = deduplicate_by(paper_auth_pairs_LH, "last author")
claims_LH = authors_claims[authors_claims['Authorship'] == 'Leading']


# %%
# create merge columns: lowercased and stripped of accents
paper_auth_pairs_LH['lh_proc'] = paper_auth_pairs_LH['last author'].str.lower()
paper_auth_pairs_LH['lh_proc'] = paper_auth_pairs_LH['lh_proc'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
claims_LH['lh_proc'] = claims_LH['Name'].str.lower()
claims_LH['lh_proc'] = claims_LH['lh_proc'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
# replace ando i by ando
claims_LH['lh_proc'] = claims_LH['lh_proc'].str.replace('ando i', 'ando')

all_LH = pd.merge(claims_LH, paper_auth_pairs_LH, on='lh_proc', how='outer')
print(len(claims_LH), len(paper_auth_pairs_LH), len(all_LH))

# %%
unique_pairs = all_LH[["Name", "last author", "lh_proc"]].drop_duplicates().sort_values("last author", ascending=True)
for i in range(0, len(unique_pairs)):
    if pd.isna(unique_pairs.iloc[i]['last author']) or pd.isna(unique_pairs.iloc[i]['Name']):
        print('💥 ', end='')
    print(f"{unique_pairs.iloc[i]['lh_proc']:<20} {unique_pairs.iloc[i]['last author']:<20}  {unique_pairs.iloc[i]['Name']}")

# %%
all_LH_inner = pd.merge(claims_LH, paper_auth_pairs_LH, on='lh_proc', how='inner')
print(len(all_LH_inner))

# %% [markdown]
# ## Let's go for last authors first

# %%
all_LH_inner

# %%
unique_leading_author

# %%

# %%
paper_auth_pairs

# %%

leading_authors_df = pd.merge(unique_leading_author, grouped_authors_info.drop(columns=['Sex']), 
left_on='Name', 
right_on='last author', how='left').drop(columns=['last author', 'first author'])

# %%
leading_authors_df

# %%

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
