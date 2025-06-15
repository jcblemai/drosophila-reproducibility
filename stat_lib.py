from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import Table2x2
import scipy.stats as stats
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
import bambi as bmb
import matplotlib.pyplot as plt

# Dataset covariate summary
def analyze_covariates(df):
    """
    Analyze covariates in the dataset and provide detailed summary
    """
    print(f"Dataset has {len(df)} rows\n")
    
        
    for col in df.columns:
        missing_count = df[col].isna().sum()
        # Calculate unique values
        unique_values = df[col].dropna().unique()
        n_unique = len(unique_values)
        # Print column info
        if n_unique <= 5:
            # Show actual values if 5 or fewer
            values_str = ", ".join([str(v) for v in sorted(unique_values)])
            print(f"  - {col:<30}: ({missing_count:<3} missing, {n_unique:<3} unique) {values_str} ")
        else:
            # Show count if more than 5
            print(f"  - {col:<30}: ({missing_count:<3} missing, {n_unique:<3} unique)")


def report_proportion(successes, total, confidence=0.95, end_sentence="of tested claims were irreproducible."):
    """
    Report a proportion with a Wilson score confidence interval.
    
    Args:
        successes (int): Number of successes/events (e.g., challenged claims).
        total (int): Total number of trials (e.g., total claims).
        confidence (float): Confidence level, default 0.95.
        
    Returns:
        str: Formatted sentence.
    """
    prop = successes / total
    ci_low, ci_upp = proportion_confint(count=successes, nobs=total, alpha=1-confidence, method='wilson')
    
    prop_percent = 100 * prop
    ci_low_percent = 100 * ci_low
    ci_upp_percent = 100 * ci_upp
    
    return f"{prop_percent:.1f}% (95% CI: {ci_low_percent:.1f}%–{ci_upp_percent:.1f}%) {end_sentence}"


def report_categorical_comparison(var_grouped, labels, outcome='Challenged', alpha=0.05, what_str=""):
    """
    Full report comparing two groups: proportions with CI, OR with CI, and Fisher p-value.
    
    Args:
        var_grouped (DataFrame): Table with counts per group.
        labels (list): Names of the two groups [group1, group2].
        outcome (str): Column for outcome count, e.g., 'Challenged'.
        alpha (float): Significance level (default 0.05).
        
    Returns:
        str: Formatted professional sentence.
        dict: Summary results for export.
    """
    group1, group2 = labels
    group1_challenged = var_grouped.loc[group1, f'{outcome}']
    group1_total = var_grouped.loc[group1, 'Major claims']
    group2_challenged = var_grouped.loc[group2, f'{outcome}']
    group2_total = var_grouped.loc[group2, 'Major claims']
    
    # Contingency table
    table = [[group1_challenged, group1_total - group1_challenged],
             [group2_challenged, group2_total - group2_challenged]]
    
    # Wilson confidence intervals
    group1_prop = group1_challenged / group1_total
    group2_prop = group2_challenged / group2_total
    ci1_low, ci1_upp = proportion_confint(group1_challenged, group1_total, alpha=1-alpha, method='wilson')
    ci2_low, ci2_upp = proportion_confint(group2_challenged, group2_total, alpha=1-alpha, method='wilson')
    
    # Odds Ratio and Fisher p-value
    ct = Table2x2(table)
    or_estimate = ct.oddsratio
    ci_low, ci_upp = ct.oddsratio_confint()
    _, p_value = stats.fisher_exact(table)
    
    # If OR < 1, flip groups to make OR > 1 for clarity (optional)
    if or_estimate < 1:
        group1, group2 = group2, group1
        group1_prop, group2_prop = group2_prop, group1_prop
        ci1_low, ci1_upp, ci2_low, ci2_upp = ci2_low, ci2_upp, ci1_low, ci1_upp
        or_estimate = 1 / or_estimate
        ci_low, ci_upp = 1/ci_upp, 1/ci_low

    significance = "not significantly associated with" if p_value > alpha else "significantly associated with"
    group1_str = str(labels[0]).lower()
    group2_str = str(labels[1]).lower()
    sentence = (f"{what_str} {group1_str} vs {group2_str} was {significance} claim reproducibility "
                f"(p = {p_value:.2f});\n  {group1_prop*100:.1f}% (95% CI {ci1_low*100:.1f}–{ci1_upp*100:.1f}%) vs "
                f"{group2_prop*100:.1f}% (95% CI {ci2_low*100:.1f}–{ci2_upp*100:.1f}%) of claims were challenged "
                f"for {group1} vs {group2}, respectively. \n Odds Ratio = {or_estimate:.2f} (95% CI {ci_low:.2f}–{ci_upp:.2f}).")
    
    # Return sentence and dictionary for export
    summary = {
        'Group1': group1,
        'Group2': group2,
        'Group1 Challenged % (95% CI)': f"{group1_prop*100:.1f}% ({ci1_low*100:.1f}–{ci1_upp*100:.1f}%)",
        'Group2 Challenged % (95% CI)': f"{group2_prop*100:.1f}% ({ci2_low*100:.1f}–{ci2_upp*100:.1f}%)",
        'Odds Ratio': round(or_estimate, 2),
        'OR 95% CI': f"{ci_low:.2f}–{ci_upp:.2f}",
        'Fisher p-value': round(p_value, 3),
        'Significance': significance
    }
    
    return sentence, summary


def tidy_glm_or_table(glm_res, drop_intercept=True, digits=2):
    """
    Return a dataframe of coefficients, ORs, CIs and p-values.

    Parameters
    ----------
    glm_res : statsmodels GLMResults
    drop_intercept : bool
    digits : int        # rounding for display

    Returns
    -------
    pd.DataFrame
    """
    table = (glm_res.params
             .to_frame("coef")
             .join(glm_res.conf_int().rename(columns={0: "ci_low", 1: "ci_high"}))
             .join(glm_res.pvalues.to_frame("pval")))

    table["OR"]       = np.exp(table["coef"])
    table["OR_low"]   = np.exp(table["ci_low"])
    table["OR_high"]  = np.exp(table["ci_high"])

    if drop_intercept:
        table = table.loc[~table.index.str.contains("Intercept")]

    return (table[["OR","OR_low","OR_high","pval"]]
            .round(digits)
            .sort_values("OR", ascending=False))


def forest_plot(glm_res, title="Adjusted odds ratios", ax=None,
                figsize=None, drop_intercept=True, c="black"):
    """
    Draw a horizontal forest plot of ORs with 95 % CIs.

    Returns
    -------
    ax  (matplotlib Axes)
    """
    tbl = tidy_glm_or_table(glm_res, drop_intercept=drop_intercept, digits=3)
    if figsize is None:
        figsize = (6, 0.5 + 0.4 * len(tbl))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(tbl))[::-1]           # top-to-bottom
    ax.hlines(y, tbl["OR_low"], tbl["OR_high"], color=c, lw=2)
    ax.scatter(tbl["OR"], y, color=c, s=45, zorder=5)

    ax.axvline(1, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels(tbl.index)
    ax.set_title(title)
    plt.tight_layout()
    return ax

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


def run_BinomialBayesMixedGLM_deprecated():
    """
    The model that did not work, see my note
    """
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

()
