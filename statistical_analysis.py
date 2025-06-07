# -*- coding: utf-8 -*-
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
import stat_lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Example usage
print(stat_lib.report_proportion(38, 45))

# %%
first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")
first_author_claims = first_author_claims[['id', 'first_author_key', 'First Author Sex', 'PhD Post-doc']]
leading_author_claims = pd.read_csv("preprocessed_data/leading_author_claims.csv")
lh_first_papers_year = pd.read_csv("preprocessed_data/lh_first_papers_year.csv", sep=";")
leading_author_claims = pd.merge(leading_author_claims, lh_first_papers_year, how="left", on="leading_author_key")
leading_author_claims = leading_author_claims[['id', 'leading_author_key', 'Historical lab after 1998', 'Continuity',  'Leading Author Sex', 'Junior Senior',"F and L",  'first_paper_year', 
'year', 'binned_year','journal_category', 'ranking_category', # Paper covariates
'assessment_type_grouped', # claim outcome
]]

# %%
all_covar = pd.merge(first_author_claims, leading_author_claims, how="left", left_on="id", right_on="id", suffixes=("", "_lh"))
all_covar = all_covar.drop(all_covar.filter(regex='_lh$').columns, axis=1)

# %%
# Let's redo the categories
all_covar

# all_covar has lenght(1006) and for columns
# - id: unique identifier for the claim
# - First author covariages
#   - first_author_key: unique identifier for the first author (289 unique)
#   - First Author Sex: Female or Male, 25 missing
#   - PhD Post-doc: PhD or Post-doc, 128 missing
# - Leading author covariates
#   - leading_author_key: unique identifier for the leading author (156 unique)
#   - Historical lab after 1998: True or False, 176 missing
#   - Continuity: True or False, 0 missing
#   - Leading Author Sex: Male or Female, 0 missing
#   - Junior Senior: Junior PI or Senior PI, 11 missing
#   - F and L: True or False, 213 missing
#   - first_paper_year: year of the first paper of the leading author, 0 missing (32 unique)
# - Paper covariates
#   - year: year of the paper, 0 missing (32 unique)
#   - impact_factor: impact factor of the journal, 0 missing (62 unique)
#   - shangai_ranking_2010: Shanghai ranking of the journal in 2010, 441 missing (36 unique)



# %%
all_covar = all_covar[[
    'year','impact_factor', 'shangai_ranking_2010', # Paper covariates
     # First author covariates
    # Leading author covariates
    
      ]]


# %%
all_covar["challenged_flag"] = all_covar["assessment_type_grouped"] == "Challenged"

# %%
# replace all space by underscore in column names
all_covar.columns = all_covar.columns.str.replace(' ', '_')

# %%

effect = [
   # ["scale", "shangai_ranking_2010"],
    ["scale", "year"],
    ["C", "Leading_Author_Sex"],

]

fixed = [f"{e[0]}({e[1]})" for e in effect]


cols = [e[1] for e in effect]

print(all_covar[cols + ['challenged_flag']].describe(include='all'))
print(all_covar[cols + ['challenged_flag']].isna().sum())


# %%
fixed

# %%

# ------------- frequentist quick check -----------------
glm_res = stat_lib.fit_glm_cluster(all_covar, fixed, cluster_cols=('first_author_key', 'leading_author_key'))
print(glm_res.summary())

# Nicely formatted OR table
or_table = (glm_res
            .params
            .apply(lambda b: (pd.Series({'OR':  np.exp(b)})))
            .join(glm_res.conf_int().apply(np.exp))
            .rename(columns={0: 'CI_low', 1: 'CI_high'}))
print("\nAdjusted Odds Ratios\n", or_table)


# %%
glm_res.summary2()

# %%
or_tbl = stat_lib.tidy_glm_or_table(glm_res)
print(or_tbl)

# 2) forest plot
stat_lib.forest_plot(glm_res, title="Figure X – Adjusted ORs for claim irreproducibility");
plt.show()

# %%
model, idata = stat_lib.fit_bayesian_mixed(all_covar, fixed)
print(model.summary(idata, hdi=0.95))

# e.g. plot forest of fixed effects
import arviz as az
az.plot_forest(idata, filter_vars="like", kind='forest', combined=True);
