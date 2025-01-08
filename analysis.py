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
else:
    dfs = utils.load_dfs(method='pickle')

# %%
import pickle
with open('dfs.pickle', 'wb') as f:
    pickle.dump(dfs, f)

# %%

# %%
import pickle
import pandas as pd
def clean_df(df):
    columns_to_remove = ['user_id', 'orcid_user_id', 
                    'created_at', 'updated_at', 'assertion_updated_at', 
                    'workspace_id', 'user_id', 'doi', 'organism_id', 'pmid', 
                    'all_tags_json', 'obsolete', 'ext', 'badge_classes','pluralize_title',
                    'can_attach_file', 'refresh_side_panel', 'icon_classes', 'btn_classes']
    patterns_to_remove = ['validated', 'filename', 'obsolete_article']
    for col in columns_to_remove:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    for pattern in patterns_to_remove:
        cols = [c for c in df.columns if pattern in c]
        df.drop(cols, axis=1, inplace=True)
    return df

# load it back with:
with open('dfs.pickle', 'rb') as f:
    dfs = pickle.load(f)


# preprocess: my unit here is the claims that are in assertion
# check wich columns have _id
claims = clean_df(dfs["assertions"])
id_cols = [col for col in claims.columns if "_id" in col]
print(id_cols)
# merge the article columsn
articles = clean_df(dfs["articles"])
articles = articles.rename(columns={"id": "article_id"})
claims = claims.merge(articles, on="article_id", how="left", suffixes=('', '_article')).drop("article_id", axis=1)


id_cols = [col for col in claims.columns if "_id" in col]
print(id_cols)

journals = clean_df(dfs["journals"])
journals = journals.drop('tag', axis=1).rename(columns={"id": "journal_id", "name": "journal_name"})
claims = claims.merge(journals, on="journal_id", how="left", suffixes=('', '_journal')).drop("journal_id", axis=1)

# same for assertion_type
assertion_types = clean_df(dfs["assertion_types"])
assertion_types = assertion_types.rename(columns={"id": "assertion_type_id", "name": "assertion_type"})
claims = claims.merge(assertion_types, on="assertion_type_id", how="left", suffixes=('', '_assertion_type')).drop("assertion_type_id", axis=1)

# same for assessment_type_id
assessment_types = clean_df(dfs["assessment_types"])
assessment_types = assessment_types.rename(columns={"id": "assessment_type_id", "name": "assessment_type"})
claims = claims.merge(assessment_types, on="assessment_type_id", how="left", suffixes=('', '_assessment_type')).drop("assessment_type_id", axis=1)

id_cols = [col for col in claims.columns if "_id" in col]
print(id_cols)

# %%
claims = claims.drop(['published_at', 'badge_tag_classes','description', 'additional_context', 'references_txt'], axis=1) # most not consistently used accross dataset
claims = claims.set_index('id')
claims.to_csv('claims.csv')


# %%

def truncate_string(s, max_length=100):
    """Truncate string to max_length characters."""
    if isinstance(s, str) and len(s) > max_length:
        return s[:max_length] + '...'
    return s

# for LLMs
df_truncated = claims.copy()
for col in string_columns:
    if col in df_truncated.columns:
        df_truncated[col] = df_truncated[col].apply(lambda x: truncate_string(x))

# Save truncated dataframe
df_truncated.to_csv('claims_truncated.csv', index=False)


# %%
claims.head(40)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('claims.csv')

# Filter for major claims
df_major = df[df['assertion_type'] == 'major_claim']

# Create journal type categories
def categorize_journal(row):
    if row['journal_name'] in ['Nature', 'Science', 'Cell']:
        return 'Trophy'
    elif row['impact_factor'] >= 10:
        return 'High Impact'
    else:
        return 'Low Impact'

df_major['journal_type'] = df_major.apply(categorize_journal, axis=1)

# Calculate percentages
grouped = pd.crosstab(df_major['journal_type'], 
                     df_major['assessment_type'], 
                     normalize='index') * 100

# Set up the plot style
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Create figure and axis
fig, ax = plt.subplots()

# Define colors for each assessment type (colorblind-friendly palette)
colors = {
    'Verified': '#2ecc71',      # green
    'Challenged': '#e74c3c',    # red
    'Partially verified': '#f1c40f',  # yellow
    'Unchallenged': '#95a5a6',  # gray
    'Mixed': '#9b59b6'          # purple
}

# Create the stacked bar plot
bottom = np.zeros(len(grouped.index))

for column in grouped.columns:
    values = grouped[column]
    ax.bar(grouped.index, values, bottom=bottom, label=column, 
           color=colors.get(column, '#333333'), width=0.65)
    bottom += values

# Customize the plot
ax.set_ylabel('Percentage of Claims (%)')
ax.set_xlabel('Journal Category')
ax.set_title('Distribution of Major Claims by Journal Type\nand Assessment Category', 
             pad=20)

# Add legend
ax.legend(title='Assessment Type', bbox_to_anchor=(1.05, 1), 
         loc='upper left', frameon=True)

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add sample sizes at the bottom of each bar
for i, journal_type in enumerate(grouped.index):
    count = len(df_major[df_major['journal_type'] == journal_type])
    ax.text(i, -5, f'n = {count}', ha='center', va='top')

# Extend y-axis slightly to accommodate sample size labels
ax.set_ylim(-10, 105)

# Save the figure with high resolution
plt.savefig('journal_claims_distribution.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none')
plt.savefig('journal_claims_distribution.pdf', 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none')

# Display counts
counts = pd.crosstab(df_major['journal_type'], df_major['assessment_type'])
print("\nRaw counts:")
print(counts)

# Display percentages
print("\nPercentages:")
print(grouped.round(1))

# %%



# %%
claims = clean_df(claims)
claims.c

# %%
claims.columns

# %%


