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
# Example usage
print(stat_lib.report_proportion(38, 45))

# %%
first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")
leading_author_claims = pd.read_csv("preprocessed_data/leading_author_claims.csv")

# %%
all_covar = pd.merge(first_author_claims, leading_author_claims, how="left", left_on="id", right_on="id", suffixes=("", "_lh"))
all_covar = all_covar.drop(all_covar.filter(regex='_lh$').columns, axis=1)

all_covar.columns

# %%
all_covar["challenged_flag"] = all_covar["assessment_type_grouped"] == "Challenged"

# %%

effect = [
    #["scale", "shangai_ranking_2010_lh"],
    ["scale", "year"],
    ["C", "Sex"],
    ["C", "sex"],
]

fixed = [f"{e[0]}({e[1]})" for e in effect]


cols = [e[1] for e in effect]

print(all_covar[cols + ['challenged_flag']].describe(include='all'))
print(all_covar[cols + ['challenged_flag']].isna().sum())


# %%

# ------------- frequentist quick check -----------------
glm_res = stat_lib.fit_glm_cluster(all_covar, fixed, cluster_cols=('author_key_lh', 'author_key_fh'))
print(glm_res.summary())
import numpy as np
# Nicely formatted OR table
or_table = (glm_res
            .params
            .apply(lambda b: (pd.Series({'OR':  np.exp(b)})))
            .join(glm_res.conf_int().apply(np.exp))
            .rename(columns={0: 'CI_low', 1: 'CI_high'}))
print("\nAdjusted Odds Ratios\n", or_table)

# ------------- full Bayesian mixed model ---------------
model, idata = stat_lib.fit_bayesian_mixed(all_covar, fixed)
print(model.summary(idata, hdi=0.95))

# e.g. plot forest of fixed effects
import arviz as az
az.plot_forest(idata, filter_vars="like", kind='forest', combined=True);
