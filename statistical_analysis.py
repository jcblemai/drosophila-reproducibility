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
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import stat_lib
import pytensor, os
pytensor.config.cxx = "/usr/bin/clang++"
os.environ["CXXFLAGS"] = "-std=c++17"
import bambi as bmb
import patsy
import arviz as az


# %%
first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")


leading_author_claims = pd.read_csv("preprocessed_data/leading_author_claims.csv")
lh_first_papers_year = pd.read_csv("preprocessed_data/lh_first_papers_year.csv", sep=";")
leading_author_claims = pd.merge(leading_author_claims, lh_first_papers_year, how="left", on="leading_author_key")

first_author_claims = first_author_claims[[
                                            'id', 
                                            'first_author_key',
                                            'First Author Sex',
                                            'PhD Post-doc',
                                            'First Author Become a PI'
                                            ]]

leading_author_claims = leading_author_claims[[
                                            'id',  
                                            'leading_author_key', 
                                            'Historical lab after 1998', 
                                            'Continuity',
                                            'Leading Author Sex',
                                            'Junior Senior',
                                            "F and L",
                                            'first_lh_or_fh_paper_year', 
                                            #  ~~~ Paper covariates
                                            'year', 
                                            'year_binned',
                                            'journal_category',
                                            'ranking_category', 
                                            # ~~~ outcome variables
                                            'assessment_type_grouped',
                                            ]]

leading_author_claims["first_paper_before_1995"] = leading_author_claims["first_lh_or_fh_paper_year"] < 1995

# %%
all_covar = pd.merge(first_author_claims, leading_author_claims, how="left", left_on="id", right_on="id", suffixes=("", "_lh"))
all_covar = all_covar.drop(all_covar.filter(regex='_lh$').columns, axis=1)
all_covar["challenged_flag"] = (all_covar["assessment_type_grouped"] == "Challenged").astype(int)

all_covar.columns = all_covar.columns.str.replace(' ', '_', regex=False)
all_covar.columns = all_covar.columns.str.replace('-', '', regex=False)

stat_lib.analyze_covariates(all_covar)

# %%

df = all_covar.copy()

# --- Spline for year (3 knots to cut collinearity) ---
year_splines = patsy.bs(df['year'], df=3, include_intercept=False)
# patsy.bs returns a design‐matrix; we add columns directly:
df[['year_s1','year_s2','year_s3']] = pd.DataFrame(year_splines, index=df.index)

df['challenged_flag'] = df['assessment_type_grouped'].eq('Challenged').astype(int) 


# %%

# Bambi syntax – common (fixed) effects + group-specific intercepts
formula = (
    "challenged_flag ~ "
    "C(journal_category, Treatment('Low Impact')) + "
    "year_s1 + year_s2 + year_s3 + "
    "C(ranking_category, Treatment('Not Ranked')) + "
    "C(First_Author_Sex, Treatment('Male')) + C(PhD_Postdoc, Treatment('PhD')) + "
    "C(Leading_Author_Sex, Treatment('Male')) + C(Junior_Senior, Treatment('Senior PI')) + "
    "C(Continuity, Treatment(False)) + C(first_paper_before_1995, Treatment(False)) + "
    "(1|first_author_key) + (1|leading_author_key)"
)
# Set informative priors
tight_prior = bmb.Prior("Normal", mu=0, sigma=1)

# Calculate informative intercept prior based on observed challenge rate
pi = df.challenged_flag.mean()
priors = { 
    "year_s1": tight_prior, 
    "year_s2": tight_prior, 
    "year_s3": tight_prior,
    "Intercept": bmb.Prior("Normal", mu=np.log(pi/(1-pi)), sigma=1.5) #does not changed much.
}
# Weakly-informative Normal(0,2.5) priors on all betas (Bambi default is OK)
model = bmb.Model(formula, df, family="bernoulli", dropna=True)

# Full NUTS sampling
idata = model.fit(draws=2000, 
                tune=1000, 
                chains=4, 
                cores=4, 
                target_accept=0.9, 
                random_seed=123, 
                priors=priors,
                idata_kwargs={"log_likelihood": True}) # for LOO, Leave-One-Out cross-validation, which is probably only useful for model comparison, not for this analysis.
# Do predictions for PPC
model.predict(idata, kind="response")

#%%
# ── Convergence ───────────────────────────────────
summary_stats = az.summary(idata, round_to=2)          # R-hat, ESS
# TODO check ESS > 2000 and R-hat < 1.01
print(f"min ESS: {summary_stats['ess_bulk'].min():.1f}, max ESS: {summary_stats['ess_bulk'].max():.1f}")
print(f"min R-hat: {summary_stats['r_hat'].min():.3f}, max R-hat: {summary_stats['r_hat'].max():.3f}")

loo = az.loo(idata, pointwise=True)
print(loo)

#print(summary_stats)
#az.plot_trace(idata, figsize=(8, 20));

#%%
# ── Posterior-predictive check ────────────────────
ppc_samples = idata.posterior_predictive["challenged_flag"]
posterior_proportions = ppc_samples.mean(dim=["__obs__"])
az.plot_posterior(posterior_proportions, hdi_prob=0.94)

# Calculate means
observed_mean = df['challenged_flag'].mean()

plt.figure(figsize=(8, 5))
plt.hist(posterior_proportions.values.flatten(), bins=30, alpha=0.7, density=True, color='skyblue', label='PPC samples')
plt.axvline(observed_mean, color='red', linewidth=2, label=f'Observed mean: {observed_mean:.3f}')
plt.xlabel('Mean challenged rate')
plt.ylabel('Density')
plt.title('Posterior Predictive Check - Mean Challenged Rate')
plt.legend()
plt.tight_layout()
plt.show()
print(f"PPC completed successfully. Observed mean: {observed_mean:.3f}")

# %%
# ── ODDS RATIOS BY EFFECT CATEGORIES ────────────────────
print("\n=== ODDS RATIOS ANALYSIS ===")
posterior = idata.posterior
fixed_vars = [var for var in idata.posterior.data_vars if not ('_sigma' in var or '_offset' in var)]
coef = az.summary(posterior, var_names=fixed_vars, kind='stats')
coef['OR'] = np.exp(coef['mean'])
coef['OR_low'] = np.exp(coef['hdi_3%'])
coef['OR_high'] = np.exp(coef['hdi_97%'])

# Categorize effects
first_author_effects = coef[coef.index.str.contains('First_Author_Sex|PhD_Postdoc')]
leading_author_effects = coef[coef.index.str.contains('Leading_Author_Sex|Junior_Senior|Continuity|first_paper_before_1995')]
paper_effects = coef[coef.index.str.contains('journal_category|ranking_category|year_s')]
intercept_effect = coef[coef.index.str.contains('Intercept')]

# Display tables
print("\n--- First Author Effects ---")
print(first_author_effects[['OR', 'OR_low', 'OR_high']].round(3))

print("\n--- Leading Author Effects ---")
print(leading_author_effects[['OR', 'OR_low', 'OR_high']].round(3))

print("\n--- Paper/Journal Effects ---")
print(paper_effects[['OR', 'OR_low', 'OR_high']].round(3))

stat_lib.create_forest_plot(first_author_effects, "First Author Effects", 'blue')
stat_lib.create_forest_plot(leading_author_effects, "Leading Author Effects", 'green') 
stat_lib.create_forest_plot(paper_effects, "Paper/Journal Effects", 'orange')


# %%
# ── RANDOM EFFECTS PLOTS ────────────────────
# Find all random effect variables
print("Available random effect variables:")
re_vars = [var for var in posterior.data_vars if ('first_author_key' in var or 'leading_author_key' in var)]
for var in re_vars:
    print(f"  - {var}")

# First Author Random Effects
fa_vars = {"First":'1|first_author_key', "Leading":'1|leading_author_key'}

for var_type, fa_var in fa_vars.items():
    print(f"\n--- {var_type} Author Random Effects ---")
    # Summary statistics for all var_type author effects
    fa_summary = az.summary(posterior, var_names=[fa_var])
            
    # Plot distribution of random effects
    plt.figure(figsize=(12, 6))

    # Left panel: Distribution of random effects
    plt.subplot(1, 2, 1)
    fa_means = fa_summary['mean'].values
    plt.hist(fa_means, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero effect')
    plt.axvline(fa_means.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean: {fa_means.mean():.3f}')
    plt.xlabel('Random Effect (log odds)')
    plt.ylabel('Count')
    plt.title('Distribution of {var_type} Author Random Effects')
    plt.legend()
    plt.grid(True, alpha=0.3)
            
    # Right panel: Caterpillar plot (top/bottom effects)
    plt.subplot(1, 2, 2)
    fa_summary_sorted = fa_summary.sort_values('mean')

    # Show top and bottom 15 effects
    n_show = min(15, len(fa_summary_sorted))
    top_effects = fa_summary_sorted.tail(n_show)
    bottom_effects = fa_summary_sorted.head(n_show)
    combined_effects = pd.concat([bottom_effects, top_effects])

    y_pos = range(len(combined_effects))
    plt.errorbar(combined_effects['mean'], y_pos,
                xerr=[combined_effects['mean'] - combined_effects['hdi_3%'],
                        combined_effects['hdi_97%'] - combined_effects['mean']],
                fmt='o', capsize=3, color='blue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)

    # Clean up labels
    labels = []
    for idx in combined_effects.index:
        if '[' in idx and ']' in idx:
            label = idx.split('[')[1].split(']')[0]
        else:
            label = idx
        labels.append(label[:20])  # Truncate long names

    plt.yticks(y_pos, labels)
    plt.xlabel('Random Effect (log odds)')
    plt.title(f'Top/Bottom {n_show} {var_type} Author Effects')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"{var_type} author random effects summary:")
    print(f"  Number of authors: {len(fa_means)}")
    print(f"  Mean effect: {fa_means.mean():.3f}")
    print(f"  SD of effects: {fa_means.std():.3f}")
    print(f"  Range: [{fa_means.min():.3f}, {fa_means.max():.3f}]")

    

# %%
# ── NEW MODEL: PREDICTING FIRST AUTHOR BECOMING PI ────────────────────

# Examine the data structure
print("All_covar columns and data types:")
print(all_covar.dtypes)
print("\nAll_covar shape:", all_covar.shape)
print("\nUnique values in key columns:")
print("Journal categories:", all_covar['journal_category'].unique())
print("Ranking categories:", all_covar['ranking_category'].unique())
print("First Author Sex:", all_covar['First_Author_Sex'].unique())
print("PhD Postdoc:", all_covar['PhD_Postdoc'].unique())
print("First Author Become a PI:", all_covar['First_Author_Become_a_PI'].unique())

# Create author-level dataframe with highest impact paper and ranking for each author
print("\nCreating author-level dataframe...")

# Define journal impact hierarchy (higher value = higher impact)
journal_impact_map = {
    'Trophy Journals': 3,
    'High Impact': 2, 
    'Low Impact': 1
}

# Define ranking hierarchy (higher value = better ranking)
ranking_map = {
    'Top 50': 4,
    '51-100': 3,
    '101+': 2,
    'Not Ranked': 1
}

# Add numeric columns for aggregation
all_covar_temp = all_covar.copy()
all_covar_temp['journal_impact_score'] = all_covar_temp['journal_category'].map(journal_impact_map)
all_covar_temp['ranking_score'] = all_covar_temp['ranking_category'].map(ranking_map)

# Group by first_author_key to get one row per author
author_df = all_covar_temp.groupby('first_author_key').agg({
    'First_Author_Sex': 'first',  # Take first occurrence (should be same for all papers by same author)
    'PhD_Postdoc': 'first',
    'First_Author_Become_a_PI': 'first',
    'journal_impact_score': 'max',  # Highest impact journal
    'ranking_score': 'max',  # Highest ranking institution
    'id': 'count'  # Count number of papers per author
}).reset_index()

# Rename the count column and add log transformation
author_df = author_df.rename(columns={'id': 'num_papers'})
author_df['log_num_papers'] = np.log(author_df['num_papers'])

# Map back to categorical labels for highest impact/ranking
reverse_journal_map = {v: k for k, v in journal_impact_map.items()}
reverse_ranking_map = {v: k for k, v in ranking_map.items()}

author_df['highest_impact_journal'] = author_df['journal_impact_score'].map(reverse_journal_map)
author_df['highest_ranking_institution'] = author_df['ranking_score'].map(reverse_ranking_map)

# Drop the numeric scores, keep only the categorical labels
author_df = author_df.drop(['journal_impact_score', 'ranking_score'], axis=1)

print(f"Created author-level dataframe with {len(author_df)} authors")
print("Sample of author_df:")
print(author_df.head())

print("\nDistribution of outcomes:")
print(author_df['First_Author_Become_a_PI'].value_counts())
print("\nDistribution of highest impact journals:")
print(author_df['highest_impact_journal'].value_counts())
print("\nDistribution of highest ranking institutions:")
print(author_df['highest_ranking_institution'].value_counts())
print("\nNumber of papers per author statistics:")
print(f"Mean: {author_df['num_papers'].mean():.2f}")
print(f"Median: {author_df['num_papers'].median():.2f}")
print(f"Range: {author_df['num_papers'].min()}-{author_df['num_papers'].max()}")
print(f"Log(papers) mean: {author_df['log_num_papers'].mean():.2f}")
print(f"Log(papers) std: {author_df['log_num_papers'].std():.2f}")

# Build logistic regression model for predicting PI status
print("\n=== BUILDING LOGISTIC REGRESSION MODEL ===")

# Clean data - remove rows with missing values
author_df_clean = author_df.dropna()
print(f"After removing missing values: {len(author_df_clean)} authors")

# Convert outcome to binary numeric
author_df_clean['become_pi_binary'] = (author_df_clean['First_Author_Become_a_PI'] == True).astype(int)

# Build the model formula
pi_formula = (
    "become_pi_binary ~ "
    "C(First_Author_Sex, Treatment('Male')) + "
    "C(PhD_Postdoc, Treatment('PhD')) + "
    "C(highest_impact_journal, Treatment('Low Impact')) + "
    "C(highest_ranking_institution, Treatment('Not Ranked')) + "
    "log_num_papers"
)

# Set priors
pi_rate = author_df_clean['become_pi_binary'].mean()
pi_priors = {
    "Intercept": bmb.Prior("Normal", mu=np.log(pi_rate/(1-pi_rate)), sigma=1.5),
    "log_num_papers": bmb.Prior("Normal", mu=0, sigma=1)
}

print(f"Building model with {len(author_df_clean)} authors")
print(f"PI rate: {pi_rate:.3f}")

# Fit the model
pi_model = bmb.Model(pi_formula, author_df_clean, family="bernoulli", dropna=True)

# Sample from posterior
pi_idata = pi_model.fit(draws=2000, 
                       tune=1000, 
                       chains=4, 
                       cores=4, 
                       target_accept=0.9, 
                       random_seed=123, 
                       priors=pi_priors,
                       idata_kwargs={"log_likelihood": True})

# Do predictions for PPC
pi_model.predict(pi_idata, kind="response")

print("Model fitting completed successfully!")

# Model diagnostics
print("\n=== MODEL DIAGNOSTICS ===")
pi_summary_stats = az.summary(pi_idata, round_to=2)
print(f"min ESS: {pi_summary_stats['ess_bulk'].min():.1f}, max ESS: {pi_summary_stats['ess_bulk'].max():.1f}")
print(f"min R-hat: {pi_summary_stats['r_hat'].min():.3f}, max R-hat: {pi_summary_stats['r_hat'].max():.3f}")

# Posterior predictive check
print("\n=== POSTERIOR PREDICTIVE CHECK ===")
pi_ppc_samples = pi_idata.posterior_predictive["become_pi_binary"]
pi_posterior_proportions = pi_ppc_samples.mean(dim=["__obs__"])

pi_observed_mean = author_df_clean['become_pi_binary'].mean()
plt.figure(figsize=(8, 5))
plt.hist(pi_posterior_proportions.values.flatten(), bins=30, alpha=0.7, density=True, color='skyblue', label='PPC samples')
plt.axvline(pi_observed_mean, color='red', linewidth=2, label=f'Observed PI rate: {pi_observed_mean:.3f}')
plt.xlabel('Mean PI rate')
plt.ylabel('Density')
plt.title('Posterior Predictive Check - Mean PI Rate')
plt.legend()
plt.tight_layout()
plt.show()

# Results analysis
print("\n=== ODDS RATIOS FOR PI PREDICTION ===")
pi_posterior = pi_idata.posterior
pi_fixed_vars = [var for var in pi_idata.posterior.data_vars if not ('_sigma' in var or '_offset' in var)]
pi_coef = az.summary(pi_posterior, var_names=pi_fixed_vars, kind='stats')
pi_coef['OR'] = np.exp(pi_coef['mean'])
pi_coef['OR_low'] = np.exp(pi_coef['hdi_3%'])
pi_coef['OR_high'] = np.exp(pi_coef['hdi_97%'])

print("Odds Ratios (95% HDI):")
print(pi_coef[['OR', 'OR_low', 'OR_high']].round(3))

# Create forest plot for PI model
stat_lib.create_forest_plot(pi_coef, "Predictors of Becoming a PI", 'purple')

print("\n=== MODEL SUMMARY ===")
print(f"Total authors analyzed: {len(author_df_clean)}")
print(f"Authors who became PIs: {author_df_clean['become_pi_binary'].sum()} ({pi_observed_mean:.1%})")
print("Model successfully predicts first author likelihood of becoming a PI based on:")
print("- Sex, PhD/Postdoc status, highest impact journal, highest ranking institution, log(number of papers)")

# %%
# ── SIMPLIFIED MODEL: JOURNAL VS UNIVERSITY RANKING EFFECTS ────────────────────

print("\n=== BUILDING SIMPLIFIED JOURNAL VS UNIVERSITY RANKING MODEL ===")

# Use the original paper-level data for this analysis
df_simple = all_covar.copy()
df_simple['challenged_flag'] = df_simple['assessment_type_grouped'].eq('Challenged').astype(int) 

print(f"Building simplified model with {len(df_simple)} papers")
print(f"Challenge rate: {df_simple['challenged_flag'].mean():.3f}")

# Simple model formula focusing on journal vs university ranking
simple_formula = (
    "challenged_flag ~ "
    "C(ranking_category, Treatment('Not Ranked')) + "
    "C(journal_category, Treatment('Low Impact')) + "
    "(1|leading_author_key)"
)

# Set priors for simple model
simple_pi = df_simple.challenged_flag.mean()
simple_priors = { 
    "Intercept": bmb.Prior("Normal", mu=np.log(simple_pi/(1-simple_pi)), sigma=1.5)
}

# Fit the simplified model
simple_model = bmb.Model(simple_formula, df_simple, family="bernoulli", dropna=True)

simple_idata = simple_model.fit(draws=2000, 
                               tune=1000, 
                               chains=4, 
                               cores=4, 
                               target_accept=0.9, 
                               random_seed=123, 
                               priors=simple_priors,
                               idata_kwargs={"log_likelihood": True})

# Do predictions
simple_model.predict(simple_idata, kind="response")

print("Simplified model fitting completed successfully!")

# Model diagnostics
print("\n=== SIMPLIFIED MODEL DIAGNOSTICS ===")
simple_summary_stats = az.summary(simple_idata, round_to=2)
print(f"min ESS: {simple_summary_stats['ess_bulk'].min():.1f}, max ESS: {simple_summary_stats['ess_bulk'].max():.1f}")
print(f"min R-hat: {simple_summary_stats['r_hat'].min():.3f}, max R-hat: {simple_summary_stats['r_hat'].max():.3f}")

# Results analysis
print("\n=== JOURNAL VS UNIVERSITY RANKING ODDS RATIOS ===")
simple_posterior = simple_idata.posterior
simple_fixed_vars = [var for var in simple_idata.posterior.data_vars if not ('_sigma' in var or '_offset' in var or 'leading_author_key' in var)]
simple_coef = az.summary(simple_posterior, var_names=simple_fixed_vars, kind='stats')
simple_coef['OR'] = np.exp(simple_coef['mean'])
simple_coef['OR_low'] = np.exp(simple_coef['hdi_3%'])
simple_coef['OR_high'] = np.exp(simple_coef['hdi_97%'])

print("Odds Ratios (95% HDI) - Journal vs University Effects:")
print(simple_coef[['OR', 'OR_low', 'OR_high']].round(3))

# Separate journal and university effects
journal_effects = simple_coef[simple_coef.index.str.contains('journal_category')]
university_effects = simple_coef[simple_coef.index.str.contains('ranking_category')]

print("\n--- Journal Category Effects ---")
if len(journal_effects) > 0:
    print(journal_effects[['OR', 'OR_low', 'OR_high']].round(3))
    stat_lib.create_forest_plot(journal_effects, "Journal Category Effects", 'red')

print("\n--- University Ranking Effects ---")
if len(university_effects) > 0:
    print(university_effects[['OR', 'OR_low', 'OR_high']].round(3))
    stat_lib.create_forest_plot(university_effects, "University Ranking Effects", 'blue')

print("\n=== SIMPLIFIED MODEL SUMMARY ===")
print(f"This model isolates the effects of journal prestige vs university ranking")
print(f"on the likelihood of claims being challenged, controlling for leading author variation")
print(f"Total papers: {len(df_simple)}")
print(f"Challenged papers: {df_simple['challenged_flag'].sum()} ({simple_pi:.1%})")

# %%
