from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import Table2x2
import scipy.stats as stats
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial

# ---------- frequentist (cluster-robust) -------------
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

def fit_glm_cluster(df, fixed_effects, outcome='challenged_flag',
                    cluster_cols=('first_author_key', 'leading_author_key')):
    import numpy as np
    import pandas as pd
    import patsy
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial

    # 1. -------- collect raw variable names -----------------
    needed_cols = {outcome}
    for term in fixed_effects:
        if term.startswith('C('):
            needed_cols.add(term[2:].strip('()'))          # C(Sex_lh) -> Sex_lh
        elif term.startswith('scale('):
            needed_cols.add(term[6:].strip('()'))          # scale(year_fh) -> year_fh
        else:
            needed_cols.add(term.split(':')[0])            # plain or interaction (keep first part)

    # 2. -------- verify they exist --------------------------
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")

    # 3. -------- drop rows with NA only on those -------------
    df_clean = df.dropna(subset=needed_cols)
    if df_clean.empty:
        raise ValueError("No rows left after dropping NAs – check missingness.")

    # 4. -------- build formula & design matrices ------------
    formula = outcome + ' ~ ' + ' + '.join(fixed_effects)
    y, X = patsy.dmatrices(formula, df_clean, return_type='dataframe')

    # 5. -------- fit GLM with cluster-robust SE -------------
    clusters = (
        df_clean[list(cluster_cols)]
        .astype(str)                      # cast every element to string
        .fillna("NA")                     # replace nan -> "NA"
        .agg("__".join, axis=1)           # join into one string label
    )
    model = sm.GLM(y, X, family=Binomial()).fit(
                cov_type='cluster',
                cov_kwds={'groups': clusters})
    return model

# ---------- Bayesian hierarchical via Bambi / PyMC ----
def fit_bayesian_mixed(df, fixed_effects, outcome='challenged_flag',
                       group_first='first_author_key', group_last='leading_author_key',
                       draws=2000, tune=1000, cores=4, seed=42):
    """
    Logistic mixed model with random intercepts for first & last author.
    Requires bambi (PyMC).  Returns the fitted model and the InferenceData.
    """
    import bambi as bmb

    fixed_formula = ' + '.join(fixed_effects) if fixed_effects else '1'
    formula = f"{outcome} ~ {fixed_formula} + (1|{group_first}) + (1|{group_last})"

    model = bmb.Model(formula, df, family='bernoulli')
    idata = model.fit(draws=draws, tune=tune, cores=cores, random_seed=seed)
    return model, idata

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
    
    prop_percent = round(100 * prop)
    ci_low_percent = round(100 * ci_low)
    ci_upp_percent = round(100 * ci_upp)
    
    return f"{prop_percent}% (95% CI: {ci_low_percent}%–{ci_upp_percent}%) {end_sentence}"


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

# ---------------  stats_model_viz.py  -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
