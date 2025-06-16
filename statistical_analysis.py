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

# For elegant forest plots, install: pip install forestplot

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
# Configure matplotlib for better notebook display
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.ion()  # Turn on interactive mode for better notebook display
import matplotlib
matplotlib.use('inline')  # Use inline backend for Jupyter
import stat_lib
import pytensor, os
pytensor.config.cxx = "/usr/bin/clang++"
os.environ["CXXFLAGS"] = "-std=c++17"
import bambi as bmb
import patsy
import arviz as az

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
                                            'article_id',
                                            # ~~~ outcome variables
                                            'assessment_type_grouped',
                                            ]]

leading_author_claims["first_paper_before_1995"] = leading_author_claims["first_lh_or_fh_paper_year"] < 1995


all_covar = pd.merge(first_author_claims, leading_author_claims, how="left", left_on="id", right_on="id", suffixes=("", "_lh"))
all_covar = all_covar.drop(all_covar.filter(regex='_lh$').columns, axis=1)
all_covar["challenged_flag"] = (all_covar["assessment_type_grouped"] == "Challenged").astype(int)

all_covar.columns = all_covar.columns.str.replace(' ', '_', regex=False)
all_covar.columns = all_covar.columns.str.replace('-', '', regex=False)

stat_lib.analyze_covariates(all_covar)

# %%
# ── NEW MODEL: PREDICTING FIRST AUTHOR BECOMING PI ──────────────────

# Note that the First Author Become a PI is NA for the first authors
# who were already PIs.

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
# Define PhD/Postdoc hierarchy (higher value = higher level)
phd_postdoc_map = {
    'Post-doc': 2,
    'PhD': 1
}
# Add numeric columns for aggregation
all_covar_temp = all_covar.copy()
all_covar_temp['journal_impact_score'] = all_covar_temp['journal_category'].map(journal_impact_map)
all_covar_temp['ranking_score'] = all_covar_temp['ranking_category'].map(ranking_map)
all_covar_temp['phd_postdoc_score'] = all_covar_temp['PhD_Postdoc'].map(phd_postdoc_map)

# drop the NA values for column First Author Become a PI, so paper
# with them as first author already being a PI are not included
all_covar_temp = all_covar_temp.dropna(subset=['First_Author_Become_a_PI'])

# Group by first_author_key to get one row per author
author_df = all_covar_temp.groupby('first_author_key').agg({
    'First_Author_Sex': 'first',  # Take first occurrence (should be same for all papers by same author)
    'First_Author_Become_a_PI': 'first',
    'journal_impact_score': 'max',  # Highest impact journal
    'ranking_score': 'max',  # Highest ranking institution
    'phd_postdoc_score': 'max',  # Highest level (Post-doc > PhD)
    'challenged_flag': 'max',  # True if any claim was challenged (max of 0/1 gives 1 if any are 1)
    'article_id': 'count'  # Count number of papers for that author as first author
}).reset_index()

# Rename the count column and add log transformation
author_df = author_df.rename(columns={'article_id': 'num_papers', 'challenged_flag': 'any_challenged'})
# Convert any_challenged to boolean
author_df['any_challenged'] = author_df['any_challenged'].astype(bool)
author_df['log_num_papers'] = np.log(author_df['num_papers'])
# Standardize num_papers (mean=0, SD=1)
author_df['num_papers_std'] = (author_df['num_papers'] - author_df['num_papers'].mean()) / author_df['num_papers'].std()

# Map back to categorical labels for highest impact/ranking
reverse_journal_map = {v: k for k, v in journal_impact_map.items()}
reverse_ranking_map = {v: k for k, v in ranking_map.items()}
reverse_phd_postdoc_map = {v: k for k, v in phd_postdoc_map.items()}

author_df['highest_impact_journal'] = author_df['journal_impact_score'].map(reverse_journal_map)
author_df['highest_ranking_institution'] = author_df['ranking_score'].map(reverse_ranking_map)
author_df['highest_phd_postdoc'] = author_df['phd_postdoc_score'].map(reverse_phd_postdoc_map)

# Drop the numeric scores, keep only the categorical labels
author_df = author_df.drop(['journal_impact_score', 'ranking_score', 'phd_postdoc_score'], axis=1)

# Drop NAs befor bambi so the PPC is meaningful
author_df = author_df.dropna()

# Convert outcome to binary numeric
author_df['become_pi_binary'] = (author_df['First_Author_Become_a_PI'] == True).astype(int)
# Build the model formula
pi_formula = (
    "become_pi_binary ~ "
    "C(First_Author_Sex, Treatment('Male')) + "
    "C(highest_phd_postdoc, Treatment('PhD')) + "
    "C(highest_impact_journal, Treatment('Low Impact')) + "
    "C(highest_ranking_institution, Treatment('Not Ranked')) + "
    "C(any_challenged, Treatment(False)) + "
    "num_papers_std"  # Standardized number of papers
    #"num_papers"
    #"log_num_papers"
)

# Set priors
pi_rate = author_df['become_pi_binary'].mean()
pi_priors = {
    "Intercept": bmb.Prior("Normal", mu=np.log(pi_rate/(1-pi_rate)), sigma=1.5),
    #"num_papers_std": bmb.Prior("Normal", mu=0, sigma=1)  # Standard normal prior for standardized variable
    #"num_papers_std": bmb.Prior("Normal", mu=0, sigma=1)  # Standard normal prior for standardized variable
}
print(f"Building model with {len(author_df)} authors")
print(f"PI rate: {pi_rate:.3f}")

# Fit the model
pi_model = bmb.Model(pi_formula, author_df, family="bernoulli", dropna=True)

# Sample from posterior
pi_idata = pi_model.fit(draws=2000, 
                       tune=1000, 
                       chains=4, 
                       cores=4, 
                       target_accept=0.9, 
                       random_seed=123, 
                       idata_kwargs={"log_likelihood": True})

# Do predictions for PPC
pi_model.predict(pi_idata, kind="response")
stat_lib.check_model_convergence(short=True, idata=pi_idata)

# Posterior predictive check
print("\n=== POSTERIOR PREDICTIVE CHECK ===")
pi_ppc_samples = pi_idata.posterior_predictive["become_pi_binary"]
pi_posterior_proportions = pi_ppc_samples.mean(dim=["__obs__"])

pi_observed_mean = author_df['become_pi_binary'].mean()
plt.figure(figsize=(8, 5))
plt.hist(pi_posterior_proportions.values.flatten(), bins=30, alpha=0.7, density=True, color='skyblue', label='PPC samples')
plt.axvline(pi_observed_mean, color='red', linewidth=2, label=f'Observed PI rate: {pi_observed_mean:.3f}')
plt.xlabel('Mean PI rate')
plt.ylabel('Density')
plt.title('Posterior Predictive Check - Mean PI Rate')
plt.legend()
plt.tight_layout()
plt.show()
plt.show()  # Ensure plot displays in notebook

# Results analysis
print("\n=== ODDS RATIOS FOR PI PREDICTION ===")
pi_posterior = pi_idata.posterior
pi_fixed_vars = [var for var in pi_idata.posterior.data_vars if not ('_sigma' in var or '_offset' in var)]
# remove the p[0] … p[250] that are not coefficients at all – (per-row fitted probabilities that PyMC/Bambi
#  stores under the name p for a Bernoulli model.
pi_fixed_vars = [var for var in pi_fixed_vars if var != "p"]
pi_coef = az.summary(pi_posterior, var_names=pi_fixed_vars, kind='stats')
# Remove intercept from results
pi_coef = pi_coef[~pi_coef.index.str.contains('Intercept')]
pi_coef['OR'] = np.exp(pi_coef['mean'])
pi_coef['OR_low'] = np.exp(pi_coef['hdi_3%'])
pi_coef['OR_high'] = np.exp(pi_coef['hdi_97%'])

print("Odds Ratios (95% HDI):")
print("DEBUG: Raw variable names in pi_coef:")
for i, var_name in enumerate(pi_coef.index):
    print(f"  {i}: '{var_name}'")
    if i >= 10:  # Only show first 10
        break
print()


pi_coef_formatted = stat_lib.format_results_table(pi_coef, clean_variable_names=True)
print(pi_coef_formatted[['OR', 'OR_low', 'OR_high']])


# Create forest plot for PI model
fig, ax = stat_lib.create_forest_plot(pi_coef, "Predictors of Becoming a PI", 'purple')
plt.show()  # Ensure plot displays in notebook

# Create elegant forest plot using forestplot package
try:
    ax_elegant = stat_lib.create_elegant_forest_plot(pi_coef, "Predictors of Becoming a PI")
    plt.show()
except Exception as e:
    print(f"Could not create elegant forest plot: {e}")
    print("You may need to install forestplot: pip install forestplot")

print("\n=== MODEL SUMMARY ===")
print(f"Total authors analyzed: {len(author_df)}")
print(f"Authors who became PIs: {author_df['become_pi_binary'].sum()} ({pi_observed_mean:.1%})")

# %%
# ── MODEL1: ALL CLAIM PREDICTOR ────────────────────
df = all_covar.copy()
# --- Spline for year (3 knots to cut collinearity) ---
year_splines = patsy.bs(df['year'], df=3, include_intercept=False)
# patsy.bs returns a design‐matrix; we add columns directly:
df[['year_s1','year_s2','year_s3']] = pd.DataFrame(year_splines, index=df.index)

df['challenged_flag'] = df['assessment_type_grouped'].eq('Challenged').astype(int) 

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
model.set_priors(priors)
# Full NUTS sampling
idata = model.fit(draws=2000, 
                tune=1000, 
                chains=4, 
                cores=4, 
                target_accept=0.9, 
                random_seed=123, 
                idata_kwargs={"log_likelihood": True}) # for LOO, Leave-One-Out cross-validation, which is probably only useful for model comparison, not for this analysis.
# Do predictions for PPC
model.predict(idata, kind="response")

stat_lib.check_model_convergence(short=True, idata=idata)

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

# ── ODDS RATIOS BY EFFECT CATEGORIES ────────────────────
print("\n=== ODDS RATIOS ANALYSIS ===")
posterior = idata.posterior
fixed_vars = [var for var in idata.posterior.data_vars if not ('_sigma' in var or '_offset' in var)]
coef = az.summary(posterior, var_names=fixed_vars, kind='stats')
# Remove intercept from results
coef = coef[~coef.index.str.contains('Intercept')]
coef['OR'] = np.exp(coef['mean'])
coef['OR_low'] = np.exp(coef['hdi_3%'])
coef['OR_high'] = np.exp(coef['hdi_97%'])

# Categorize effects
first_author_effects = coef[coef.index.str.contains('First_Author_Sex|PhD_Postdoc')]
leading_author_effects = coef[coef.index.str.contains('Leading_Author_Sex|Junior_Senior|Continuity|first_paper_before_1995')]
paper_effects = coef[coef.index.str.contains('journal_category|ranking_category|year_s')]
intercept_effect = coef[coef.index.str.contains('Intercept')]

# Display tables with clean names and formatting
print("\n--- First Author Effects ---")
first_author_clean = stat_lib.clean_variable_names_for_table(first_author_effects)
first_author_formatted = stat_lib.format_results_table(first_author_clean)
print(first_author_formatted[['OR', 'OR_low', 'OR_high']])

print("\n--- Leading Author Effects ---")
leading_author_clean = stat_lib.clean_variable_names_for_table(leading_author_effects)
leading_author_formatted = stat_lib.format_results_table(leading_author_clean)
print(leading_author_formatted[['OR', 'OR_low', 'OR_high']])

print("\n--- Paper/Journal Effects ---")
paper_effects_clean = stat_lib.clean_variable_names_for_table(paper_effects)
paper_effects_formatted = stat_lib.format_results_table(paper_effects_clean)
print(paper_effects_formatted[['OR', 'OR_low', 'OR_high']])

# Create single comprehensive forest plot with all effects
all_effects = pd.concat([
    first_author_effects.assign(category='First Author'),
    leading_author_effects.assign(category='Leading Author'), 
    paper_effects.assign(category='Paper/Journal')
])

# Create the combined forest plot
fig, ax = stat_lib.create_forest_plot(all_effects, "All Model Effects: Predictors of Challenged Claims", 'navy')

# Create elegant forest plot using forestplot package
try:
    ax_elegant = stat_lib.create_elegant_forest_plot(all_effects, "All Model Effects: Predictors of Challenged Claims")
    plt.show()
except Exception as e:
    print(f"Could not create elegant forest plot: {e}")
    print("You may need to install forestplot: pip install forestplot")


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
    fa_means = fa_summary['mean'].values
    
    print(f"{var_type} author random effects summary:")
    print(f"  Number of authors: {len(fa_means)}")
    print(f"  Mean effect: {fa_means.mean():.3f}")
    print(f"  SD of effects: {fa_means.std():.3f}")
    print(f"  Range: [{fa_means.min():.3f}, {fa_means.max():.3f}]")
    
    # Create publication-friendly forest plot for random effects
    fig, ax = stat_lib.create_random_effects_forest_plot(fa_summary, f"{var_type} Author Random Effects", 'darkgreen')
    plt.show()  # Ensure plot displays in notebook
    


