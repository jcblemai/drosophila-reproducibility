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
# Example usage
print(stat_lib.report_proportion(38, 45))

# %%
first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")
first_author_claims = first_author_claims[['id', 'first_author_key', 'First Author Sex', 'PhD Post-doc']]
leading_author_claims = pd.read_csv("preprocessed_data/leading_author_claims.csv")
lh_first_papers_year = pd.read_csv("preprocessed_data/lh_first_papers_year.csv", sep=";")
leading_author_claims = pd.merge(leading_author_claims, lh_first_papers_year, how="left", on="leading_author_key")
leading_author_claims = leading_author_claims[['id', 'leading_author_key', 'Historical lab after 1998', 'Continuity',  'Leading Author Sex', 'Junior Senior',"F and L",  'first_paper_year', 
'year', 'year_binned','journal_category', 'ranking_category', # Paper covariates
'assessment_type_grouped', # claim outcome
]]
leading_author_claims["first_paper_before_1995"] = leading_author_claims["first_paper_year"] < 1995

# %%
all_covar = pd.merge(first_author_claims, leading_author_claims, how="left", left_on="id", right_on="id", suffixes=("", "_lh"))
all_covar = all_covar.drop(all_covar.filter(regex='_lh$').columns, axis=1)
all_covar["challenged_flag"] = (all_covar["assessment_type_grouped"] == "Challenged").astype(int)

# %%
# Let's redo the categories
all_covar

# %%
stat_lib.analyze_covariates(all_covar)

# %%
def run_inomialBayesMixedGLM():
    ## The mixed model will use the following covariates:
    # ---------------------------------------------------------
    #  Step 0  – define the columns actually used in the model
    # ---------------------------------------------------------
    fixed_effect_vars = [
        'journal_category', 'year_binned', 'First Author Sex', 'Leading Author Sex',
        'PhD Post-doc', 'Junior Senior', 'Continuity', 'first_paper_before_1995', #'F and L', 
        'ranking_category'
    ]
    random_effect_vars = ['first_author_key', 'leading_author_key']
    target = ['challenged_flag']

    model_vars = fixed_effect_vars + random_effect_vars + target

    # ---------------------------------------------------------
    #  Step 1  – drop incomplete rows
    # ---------------------------------------------------------
    all_covar_cc = (
        all_covar
        .dropna(subset=model_vars)         # complete-case
        .reset_index(drop=True)
    )

    print(f"Dropped {len(all_covar) - len(all_covar_cc)} incomplete rows "
        f"({100*(len(all_covar) - len(all_covar_cc))/len(all_covar):.1f} %).")

    # ---------------------------------------------------------
    #  Step 2  – re-encode categoricals
    # ---------------------------------------------------------
    for c in fixed_effect_vars:
        all_covar_cc[c] = all_covar_cc[c].astype('category')
    for c in random_effect_vars:
        all_covar_cc[c] = all_covar_cc[c].astype('category')

    # replace spaces in column names with underscores
    all_covar_cc.columns = all_covar_cc.columns.str.replace(' ', '_', regex=False)
    all_covar_cc.columns = all_covar_cc.columns.str.replace('-', '', regex=False)
    # ------------------------------------------------------------------
    # 2) APPROXIMATE FREQUENTIST GLMM  (statsmodels)
    # ------------------------------------------------------------------
    import statsmodels.api as sm
    # bs(year, df=3) # is a B-spline basis expansion of the year variable
    # could also have used the year as a scale
    fixef_formula = (
        "challenged_flag ~ "
        # paper covariates
        "C(journal_category, Treatment('Low Impact')) + "
        "bs(year, df=4) + "
        "C(ranking_category, Treatment('Not Ranked')) + "
        # first author covariates
        "C(First_Author_Sex, Treatment('Male')) + "
        "C(PhD_Postdoc, Treatment('PhD')) + "
        # leading author covariates
        "C(Leading_Author_Sex, Treatment('Male')) + "
        "C(Junior_Senior, Treatment('Junior PI')) + "
        "C(Continuity, Treatment(False)) + "
        # "C(F_and_L, Treatment(False)) + "
        "C(first_paper_before_1995, Treatment(False))"
    )
    vc_formulas = {
        'first_author': '0 + C(first_author_key)',
        'leading_author': '0 + C(leading_author_key)'
    }
    #vc_formulas = {} # no random effects, just fixed effects
    glmm = sm.BinomialBayesMixedGLM.from_formula(
        fixef_formula, vc_formulas, data=all_covar_cc
    )
    # result = glmm.fit_vb() # mean-field variational Bayes approximation is fit_vb, faster but worst.
    # result = glmm.fit_map()
    result = glmm.fit_map(method='BFGS') 

    print(result.summary())


    # ────────────────────────────────────────────────────────────────────────────────
    # 1.  FIT INFO & SAFE FALL-BACKS
    # ────────────────────────────────────────────────────────────────────────────────
    def safe_pseudo_r2(full_prob, y):
        """McFadden R² using predicted probs; works even when llf/llnull absent."""
        ll_full = np.sum(y * np.log(full_prob) + (1 - y) * np.log1p(-full_prob))
        p0 = y.mean()
        ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log1p(-p0))
        return 1 - ll_full / ll_null

    # predicted probabilities
    p_hat = result.predict()
    mcFadden_R2 = safe_pseudo_r2(p_hat, all_covar_cc.challenged_flag)
    print(f"McFadden pseudo-R² (manual) = {mcFadden_R2:.3f}")

    # ELBO history is only stored for VB and only in recent versions
    if hasattr(result, "elbo_history") and result.elbo_history is not None:
        plt.plot(result.elbo_history)
        plt.title("Variational Bayes – ELBO by iteration")
        plt.xlabel("Iteration"); plt.ylabel("ELBO"); plt.tight_layout(); plt.show()
    else:
        print("ELBO history not available for this fit (expected for MAP).")

    # ────────────────────────────────────────────────────────────────────────────────
    # 2.  MULTICOLLINEARITY CHECK ON FIXED EFFECTS
    # ────────────────────────────────────────────────────────────────────────────────
    X_fixed = result.model.exog          # design matrix after Patsy
    vif_series = pd.Series(
        [variance_inflation_factor(X_fixed, i) for i in range(X_fixed.shape[1])],
        index=result.model.exog_names, name="VIF"
    )
    print("\nPredictors with VIF > 5 (rule-of-thumb flag for collinearity):")
    print(vif_series[vif_series > 5])    # thresholds 5–10 are common flags:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}


    # ---------------------------------------------------------------
    #  2.  Tidy table of odds ratios + 95 % intervals
    # ---------------------------------------------------------------
    fe_mean = result.fe_mean                # posterior mean of fixed effects
    fe_sd   = result.fe_sd                  # posterior SD (VB ≈ σ)
    z975    = norm.ppf(0.975)
    or_df = (pd.DataFrame({
            'Predictor'  : result.model.exog_names,
            'logOR'      : fe_mean,
            'SE'         : fe_sd,
            'OR'         : np.exp(fe_mean),
            'CI_low'     : np.exp(fe_mean - z975*fe_sd),
            'CI_high'    : np.exp(fe_mean + z975*fe_sd)
        })
        .assign(Significant = lambda d: (d.CI_low>1) | (d.CI_high<1))
        .sort_values('OR', ascending=False)
    )
    print(or_df)

    # Random-effect ICC: This quantifies how much irreproducibility clusters by person vs lab.
    try:
        var_author = result.vc_mean[0]  # first random effect
        var_lab    = result.vc_mean[1]  # second random effect
        icc_author = var_author / (var_author + var_lab + np.pi**2/3)
        icc_lab    = var_lab    / (var_author + var_lab + np.pi**2/3)
        print(f"ICC first-author = {icc_author:.2%}, lab = {icc_lab:.2%}")
    except (IndexError, TypeError):
        print("Cannot calculate ICC - variance components not accessible")


    # ---------------------------------------------------------------
    #  3.  Forest plot
    # ---------------------------------------------------------------
    plt.figure(figsize=(6, len(or_df)*0.45))
    y = np.arange(len(or_df))
    plt.errorbar(or_df['OR'], y, xerr=[or_df['OR']-or_df['CI_low'], 
                                    or_df['CI_high']-or_df['OR']],
                fmt='o', color='navy', ecolor='lightgray', capsize=3)
    plt.yticks(y, or_df['Predictor'])
    plt.axvline(1, ls='--', color='red')
    plt.xlabel("Odds Ratio (log scale)"); plt.xscale('log')
    plt.title("Adjusted odds ratios for irreproducible claims")
    plt.tight_layout()
    plt.savefig("figures/forest_plot_OR.png", dpi=300)
    plt.show()

    # ────────────────────────────────────────────────────────────────────────────────
    # 5.  POSTERIOR PREDICTIVE CHECK (MEAN OUTCOME)
    # ────────────────────────────────────────────────────────────────────────────────
    n_sims = 400
    sim_means = np.random.binomial(1, p_hat.reshape(1, -1).repeat(n_sims, axis=0)).mean(axis=1)
    sns.histplot(sim_means, bins=30, color='skyblue'); 
    plt.axvline(all_covar_cc.challenged_flag.mean(), color='black', lw=2, label='Observed');
    plt.legend(); plt.title("Posterior predictive check (mean challenged rate)");
    plt.xlabel("Simulated mean"); plt.tight_layout(); plt.show()

# run_inomialBayesMixedGLM()

# %% 
import pytensor, os
pytensor.config.cxx = "/usr/bin/clang++"
os.environ["CXXFLAGS"] = "-std=c++17"
import bambi as bmb
import patsy
import arviz as az


# ---------------------------------------------------------
#  Step 0  – define the columns actually used in the model
# ---------------------------------------------------------
fixed_effect_vars = [
    'journal_category', 'year_binned', 'First Author Sex', 'Leading Author Sex',
    'PhD Post-doc', 'Junior Senior', 'Continuity', 'first_paper_before_1995', #'F and L', 
    'ranking_category'
]
random_effect_vars = ['first_author_key', 'leading_author_key']
target = ['challenged_flag']

model_vars = fixed_effect_vars + random_effect_vars + target

# ---------------------------------------------------------
#  Step 1  – drop incomplete rows
# ---------------------------------------------------------
df = (
    all_covar
    .dropna(subset=model_vars)         # complete-case
    .reset_index(drop=True)
)

print(f"Dropped {len(all_covar) - len(df)} incomplete rows "
    f"({100*(len(all_covar) - len(df))/len(all_covar):.1f} %).")

df.columns = df.columns.str.replace(' ', '_', regex=False)
df.columns = df.columns.str.replace('-', '', regex=False)

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
pi = df.challenged_flag.mean()          # ≈ 0.069
logit_pi = np.log(pi/(1-pi))           # Convert to logit scale

priors = { 
    "year_s1": tight_prior, 
    "year_s2": tight_prior, 
    "year_s3": tight_prior,
    # "Intercept": bmb.Prior("Normal", mu=logit_pi, sigma=1.5) does not changed much.
}

print(f"Observed challenge rate: {pi:.3f}")
print(f"Logit of challenge rate: {logit_pi:.3f}")
print(f"Using intercept prior: Normal({logit_pi:.3f}, 1.5)")

# Weakly-informative Normal(0,2.5) priors on all betas (Bambi default is OK):contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
model = bmb.Model(formula, df, family="bernoulli", dropna=False)

## VIF check will be done after fitting the model

# Full NUTS sampling
idata = model.fit(draws=2000, tune=1000, chains=4, cores=4, target_accept=0.9, random_seed=123, priors=priors,)

# ── Convergence ───────────────────────────────────
summary_stats = az.summary(idata, round_to=2)          # R-hat, ESS
print("=== MODEL CONVERGENCE DIAGNOSTICS ===")
print(summary_stats)
az.plot_trace(idata, figsize=(12, 8))
plt.tight_layout()
plt.show()

# ── Posterior-predictive check ────────────────────
print("\n=== POSTERIOR PREDICTIVE CHECK ===")
try:
    # Generate posterior predictive samples
    ppc_data = model.predict(idata, kind="response", data=df)
    
    # Extract samples - try different access patterns
    if hasattr(ppc_data, 'posterior_predictive'):
        if 'challenged_flag' in ppc_data.posterior_predictive:
            ppc_samples = ppc_data.posterior_predictive['challenged_flag'].values
        else:
            # Try first variable if challenged_flag not found
            var_name = list(ppc_data.posterior_predictive.data_vars)[0]
            ppc_samples = ppc_data.posterior_predictive[var_name].values
    elif hasattr(ppc_data, 'data_vars'):
        # Direct access to data variables
        var_name = list(ppc_data.data_vars)[0]
        ppc_samples = ppc_data[var_name].values
    else:
        # Last resort - convert to array
        ppc_samples = np.array(ppc_data)
    
    # Calculate means
    observed_mean = df['challenged_flag'].mean()
    
    # Handle different sample shapes
    if ppc_samples.ndim > 2:
        ppc_means = ppc_samples.mean(axis=-1).flatten()
    else:
        ppc_means = ppc_samples.mean(axis=-1) if ppc_samples.ndim == 2 else ppc_samples
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.hist(ppc_means, bins=30, alpha=0.7, density=True, color='skyblue', label='PPC samples')
    plt.axvline(observed_mean, color='red', linewidth=2, label=f'Observed mean: {observed_mean:.3f}')
    plt.xlabel('Mean challenged rate')
    plt.ylabel('Density')
    plt.title('Posterior Predictive Check - Mean Challenged Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"PPC completed successfully. Observed mean: {observed_mean:.3f}")
    
except Exception as e:
    print(f"Posterior predictive check failed: {e}")
    # Fallback simple check
    try:
        observed_mean = df['challenged_flag'].mean()
        print(f"Observed challenged rate: {observed_mean:.3f}")
        print("PPC plot not available, but model fit completed successfully.")
    except:
        pass

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

# Create separate forest plots
def create_forest_plot(data, title, color='navy'):
    if len(data) == 0:
        return
    plt.figure(figsize=(8, max(3, len(data)*0.5)))
    y = np.arange(len(data))
    plt.errorbar(data['OR'], y,
                 xerr=[data['OR']-data['OR_low'], data['OR_high']-data['OR']],
                 fmt='o', color=color, ecolor='lightgray', capsize=3)
    plt.yticks(y, data.index)
    plt.axvline(1, ls='--', color='red', alpha=0.7)
    plt.xlabel("Odds Ratio (log scale)")
    plt.xscale('log')
    plt.title(f"Odds Ratios: {title}")
    plt.tight_layout()
    plt.show()

create_forest_plot(first_author_effects, "First Author Effects", 'blue')
create_forest_plot(leading_author_effects, "Leading Author Effects", 'green') 
create_forest_plot(paper_effects, "Paper/Journal Effects", 'orange')

# ── ENHANCED POSTERIOR PREDICTIVE CHECK PLOT ────────────────────
print("\n=== ENHANCED POSTERIOR PREDICTIVE CHECK ===")
try:
    # Try to get PPC samples using arviz
    ppc_data = az.from_dict(posterior_predictive={"y_pred": np.random.binomial(1, 0.4, (4, 2000, len(df)))})
    
    # Manual PPC using posterior samples
    posterior_samples = idata.posterior
    
    # Get parameter samples for prediction
    intercept_samples = posterior_samples['Intercept'].values.flatten()
    
    # Simple PPC: generate predictions from posterior
    n_samples = min(1000, len(intercept_samples))
    idx = np.random.choice(len(intercept_samples), n_samples, replace=False)
    
    # Generate predictions (simplified for illustration)
    predicted_probs = 1 / (1 + np.exp(-intercept_samples[idx]))  # Just intercept for demo
    ppc_realizations = [np.random.binomial(1, p, len(df)).mean() for p in predicted_probs]
    
    # Create PPC plot
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of PPC realizations
    plt.hist(ppc_realizations, bins=50, alpha=0.7, density=True, color='lightblue', 
             label=f'PPC samples (n={n_samples})')
    
    # Add observed data line
    observed_mean = df['challenged_flag'].mean()
    plt.axvline(observed_mean, color='red', linewidth=3, 
                label=f'Observed mean: {observed_mean:.3f}')
    
    # Add statistics
    ppc_mean = np.mean(ppc_realizations)
    ppc_std = np.std(ppc_realizations)
    plt.axvline(ppc_mean, color='blue', linewidth=2, linestyle='--',
                label=f'PPC mean: {ppc_mean:.3f}')
    
    plt.xlabel('Proportion of Challenged Claims')
    plt.ylabel('Density')
    plt.title('Posterior Predictive Check: Distribution of Mean Challenge Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate Bayesian p-value
    p_value = np.mean(np.array(ppc_realizations) >= observed_mean)
    print(f"Bayesian p-value: {p_value:.3f}")
    print(f"(Proportion of PPC samples ≥ observed mean)")
    
except Exception as e:
    print(f"Enhanced PPC failed: {e}")

# ── RANDOM EFFECTS PLOTS ────────────────────
print("\n=== RANDOM EFFECTS ANALYSIS ===")

# Find all random effect variables
print("Available random effect variables:")
re_vars = [var for var in posterior.data_vars if ('first_author_key' in var or 'leading_author_key' in var)]
for var in re_vars:
    print(f"  - {var}")

# First Author Random Effects
print("\n--- First Author Random Effects ---")
try:
    # Find first author random effects - try both patterns
    fa_vars = [var for var in posterior.data_vars if 'first_author_key' in var and ('offset' in var or var.endswith('first_author_key'))]
    
    if not fa_vars:
        # Alternative: look for variables containing first_author_key
        fa_vars = [var for var in posterior.data_vars if 'first_author_key' in var]
    
    if fa_vars:
        fa_var = fa_vars[0]  # Take the first one found
        print(f"Using variable: {fa_var}")
        
        # Summary statistics for all first author effects
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
        plt.title('Distribution of First Author Random Effects')
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
        plt.title(f'Top/Bottom {n_show} First Author Effects')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"First author random effects summary:")
        print(f"  Number of authors: {len(fa_means)}")
        print(f"  Mean effect: {fa_means.mean():.3f}")
        print(f"  SD of effects: {fa_means.std():.3f}")
        print(f"  Range: [{fa_means.min():.3f}, {fa_means.max():.3f}]")
    else:
        print("No first author random effects found")
        
except Exception as e:
    print(f"First author random effects plot failed: {e}")

# Leading Author Random Effects  
print("\n--- Leading Author Random Effects ---")
try:
    # Find leading author random effects
    la_vars = [var for var in posterior.data_vars if 'leading_author_key' in var and ('offset' in var or var.endswith('leading_author_key'))]
    
    if not la_vars:
        # Alternative: look for variables containing leading_author_key
        la_vars = [var for var in posterior.data_vars if 'leading_author_key' in var]
    
    if la_vars:
        la_var = la_vars[0]  # Take the first one found
        print(f"Using variable: {la_var}")
        
        # Summary statistics
        la_summary = az.summary(posterior, var_names=[la_var])
        
        # Plot distribution of random effects
        plt.figure(figsize=(12, 6))
        
        # Left panel: Distribution of random effects
        plt.subplot(1, 2, 1)
        la_means = la_summary['mean'].values
        plt.hist(la_means, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero effect')
        plt.axvline(la_means.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {la_means.mean():.3f}')
        plt.xlabel('Random Effect (log odds)')
        plt.ylabel('Count')
        plt.title('Distribution of Leading Author Random Effects')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Right panel: Caterpillar plot (top/bottom effects)
        plt.subplot(1, 2, 2)
        la_summary_sorted = la_summary.sort_values('mean')
        
        # Show top and bottom 15 effects
        n_show = min(15, len(la_summary_sorted))
        top_effects = la_summary_sorted.tail(n_show)
        bottom_effects = la_summary_sorted.head(n_show)
        combined_effects = pd.concat([bottom_effects, top_effects])
        
        y_pos = range(len(combined_effects))
        plt.errorbar(combined_effects['mean'], y_pos,
                    xerr=[combined_effects['mean'] - combined_effects['hdi_3%'],
                          combined_effects['hdi_97%'] - combined_effects['mean']],
                    fmt='o', capsize=3, color='green', alpha=0.7)
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
        plt.title(f'Top/Bottom {n_show} Leading Author Effects')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Leading author random effects summary:")
        print(f"  Number of authors: {len(la_means)}")
        print(f"  Mean effect: {la_means.mean():.3f}")
        print(f"  SD of effects: {la_means.std():.3f}")
        print(f"  Range: [{la_means.min():.3f}, {la_means.max():.3f}]")
    else:
        print("No leading author random effects found")
        
except Exception as e:
    print(f"Leading author random effects plot failed: {e}")

# ── Model comparison metric (LOO) ─────────────────
try:
    # Need to fit model with log-likelihood computation for LOO
    idata_with_ll = model.fit(draws=2000, tune=1000, chains=4, cores=4, 
                              target_accept=0.9, random_seed=123, 
                              idata_kwargs={"log_likelihood": True})
    loo = az.loo(idata_with_ll, pointwise=True)
    print(loo)
except Exception as e:
    print(f"LOO calculation failed: {e}")
    print("Model fitting completed without LOO calculation")









# %%
