# %% [markdown]
# # Extract claims for the database
# 
# ## 1. Load all tables from the databases

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
if from_database:
    # Save tables to pickle
    import pickle
    with open('preprocessed_data/dfs.pickle', 'wb') as f:
        pickle.dump(dfs, f)

# %%


# %% [markdown]
# ## 2. Create dataframe claims with all the referenced table from the dataframes

# %%
import pickle
import pandas as pd
def clean_df(df):
    columns_to_remove = ['user_id', 'orcid_user_id', 
                    'created_at', 'updated_at', 'assertion_updated_at', 
                    'workspace_id', 'user_id', 'doi', 'organism_id', 
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
with open('preprocessed_data/dfs.pickle', 'rb') as f:
    dfs = pickle.load(f)


# preprocess: my unit here is the claims that are in assertion
# check wich columns have _id
claims = clean_df(dfs["assertions"])
id_cols = [col for col in claims.columns if "_id" in col]
print(id_cols)
# merge the article columsn
articles = clean_df(dfs["articles"])
articles = articles.rename(columns={"id": "article_id"})
# We keep article ID for later
claims = claims.merge(articles, on="article_id", how="left", suffixes=('', '_article'))#.drop("article_id", axis=1)


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

# %%
string_columns = claims.select_dtypes(include='object').columns
print(string_columns)

for col in string_columns:
    claims[col] = claims[col].apply(lambda x: x.replace('&amp;', '&') if isinstance(x, str) else x)

claims.to_csv('preprocessed_data/claims.csv')

# %%
def truncate_string(s, max_length=20):
    """Truncate string to max_length characters."""
    if isinstance(s, str) and len(s) > max_length:
        return s[:max_length] + '...'
    return s

string_columns = string_columns.drop(["assessment_type"])

df_truncated = claims.copy()

for col in string_columns:
    if col in df_truncated.columns:
        df_truncated[col] = df_truncated[col].apply(lambda x: truncate_string(x))

# Save truncated dataframe
df_truncated.to_csv('preprocessed_data/claims_truncated_for_llm.csv', index=False)

# %%
df_truncated


