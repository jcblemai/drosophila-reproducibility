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

# Create publication-friendly forest plots
def create_forest_plot(data, title, color='navy', category_labels=None, reference_labels=None):
    """
    Create publication-friendly forest plots.
    
    Parameters:
    -----------
    data : DataFrame with columns OR, OR_low, OR_high
    title : str, plot title
    color : str, color for points and lines
    category_labels : dict, mapping from variable names to clean category labels
    reference_labels : dict, mapping from variable names to reference category names
    """
    if len(data) == 0:
        return
    
    # Clean up variable names for publication using shared function
    clean_labels = [clean_variable_name(idx, for_plot=True) for idx in data.index]
    
    # Handle category-based grouping if category column exists
    has_categories = 'category' in data.columns
    if has_categories:
        # Sort by category for better organization
        data_sorted = data.copy()
        category_order = ['First Author', 'Leading Author', 'Paper/Journal', 'Lowest Effects', 'Highest Effects']
        data_sorted['category_num'] = data_sorted['category'].map(
            {cat: i for i, cat in enumerate(category_order)}
        ).fillna(999)
        data_sorted = data_sorted.sort_values(['category_num', 'OR'], ascending=[True, False])
        
        # Update clean_labels for sorted data and create spaced layout
        clean_labels_spaced = []
        y_positions_spaced = []
        colors_spaced = []
        data_rows = []
        
        current_category = None
        y_pos = 0
        
        # Define colors for each category
        category_colors = {
            'First Author': 'blue',
            'Leading Author': 'green', 
            'Paper/Journal': 'orange',
            'Lowest Effects': 'red',
            'Highest Effects': 'darkgreen'
        }
        
        for idx, row in data_sorted.iterrows():
            if row['category'] != current_category:
                if current_category is not None:
                    # Add spacing for category separation
                    y_pos += 1
                
                # Add category header
                clean_labels_spaced.append(f"{row['category']} Effects")
                y_positions_spaced.append(y_pos)
                colors_spaced.append('black')  # Category headers in black
                data_rows.append(None)  # No data for category headers
                y_pos += 1
                
                current_category = row['category']
            
            # Add the actual data point
            clean_label = clean_variable_name(idx, for_plot=True)
            clean_labels_spaced.append(clean_label)
            y_positions_spaced.append(y_pos)
            colors_spaced.append(category_colors.get(row['category'], color))
            data_rows.append(row)
            y_pos += 1
        
        clean_labels = clean_labels_spaced
        y_positions = y_positions_spaced
        colors = colors_spaced
        data_for_plot = data_rows
    else:
        colors = [color] * len(data)
        y_positions = np.arange(len(data))
        data_for_plot = [row for idx, row in data.iterrows()]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(clean_labels)*0.4)))
    
    # Determine plot limits to handle extreme values
    all_ors = []
    all_or_lows = []
    all_or_highs = []
    for data_row in data_for_plot:
        if data_row is not None:
            all_ors.append(data_row['OR'])
            all_or_lows.append(data_row['OR_low'])
            all_or_highs.append(data_row['OR_high'])
    
    # Set reasonable plot limits
    min_or = min(all_or_lows) if all_or_lows else 0.1
    max_or = max(all_or_highs) if all_or_highs else 10
    
    # Define extreme value thresholds
    lower_threshold = 0.001
    upper_threshold = 10
    
    # Set plot limits with some padding
    plot_min = max(min_or * 0.8, lower_threshold * 0.1)
    plot_max = min(max_or * 1.2, upper_threshold * 10)
    
    # Plot error bars and points 
    for i, (y_pos, data_row, point_color) in enumerate(zip(y_positions, data_for_plot, colors)):
        if data_row is not None:  # Skip category headers
            or_val = data_row['OR']
            or_low = data_row['OR_low']
            or_high = data_row['OR_high']
            
            # Check for extreme values
            has_extreme_low = or_low < lower_threshold
            has_extreme_high = or_high > upper_threshold
            
            # Adjust values for plotting if extreme
            plot_or_low = max(or_low, plot_min)
            plot_or_high = min(or_high, plot_max)
            plot_or = or_val
            
            # Plot the main error bar
            ax.errorbar(plot_or, y_pos,
                        xerr=[[plot_or-plot_or_low], [plot_or_high-plot_or]],
                        fmt='o', color=point_color, ecolor='lightgray', capsize=4, 
                        markersize=6, linewidth=2)
            
            # Add arrows for extreme values
            if has_extreme_low:
                # Left arrow for extremely small values
                ax.annotate('', xy=(plot_min, y_pos), xytext=(plot_min * 1.5, y_pos),
                           arrowprops=dict(arrowstyle='<-', color=point_color, lw=2))
                
            if has_extreme_high:
                # Right arrow for extremely large values  
                ax.annotate('', xy=(plot_max, y_pos), xytext=(plot_max * 0.7, y_pos),
                           arrowprops=dict(arrowstyle='->', color=point_color, lw=2))
    
    # Reference line at OR = 1
    ax.axvline(1, ls='--', color='red', alpha=0.7, linewidth=1)
    
    # Formatting - publication ready
    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, fontsize=10)
    
    # Make category headers bold
    for i, label in enumerate(clean_labels):
        if 'Effects' in label and data_for_plot[i] is None:
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_fontsize(11)
    
    ax.set_xlabel("Odds Ratio", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis limits to handle extreme values
    ax.set_xlim(plot_min, plot_max)
    
    # Remove spines and ticks for publication quality
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)  # Remove left ticks
    
    # Create legend for categories if needed
    if has_categories and len(set(data['category'])) > 1:
        legend_elements = []
        unique_categories = data['category'].unique()
        for cat in ['First Author', 'Leading Author', 'Paper/Journal', 'Lowest Effects', 'Highest Effects']:
            if cat in unique_categories:
                col = category_colors[cat]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=col, markersize=8, label=cat))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)
    
    # Add OR values as text annotations
    for i, (y_pos, data_row) in enumerate(zip(y_positions, data_for_plot)):
        if data_row is not None:  # Only annotate actual data points, not category headers
            or_val = data_row['OR']
            or_low = data_row['OR_low']
            or_high = data_row['OR_high']
            
            # Format the annotation text with extreme value indicators
            or_low_text = f"<{lower_threshold:.3f}" if or_low < lower_threshold else f"{or_low:.2f}"
            or_high_text = f">{upper_threshold:.0f}" if or_high > upper_threshold else f"{or_high:.2f}"
            
            # Position annotation text appropriately
            text_x = min(or_val * 1.1, plot_max * 0.95)
            
            ax.text(text_x, y_pos, f'{or_val:.2f}\n[{or_low_text}, {or_high_text}]', 
                    ha='left', va='center', fontsize=8, alpha=0.8)
    
    # Grid for easier reading
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Add note about extreme values if any are present
    has_any_extreme = any(
        (data_row is not None and (data_row['OR_low'] < lower_threshold or data_row['OR_high'] > upper_threshold))
        for data_row in data_for_plot
    )
    
    if has_any_extreme:
        note_text = f"Note: Arrows indicate confidence intervals extending beyond {lower_threshold:.3f} or {upper_threshold:.0f}"
        ax.text(0.5, -0.1, note_text, transform=ax.transAxes, 
                ha='center', va='top', fontsize=9, style='italic', alpha=0.7)
    
    # Adjust layout for publication quality
    plt.tight_layout()
    plt.subplots_adjust(left=0.35, right=0.85, top=0.9, bottom=0.15)  # More bottom space for note
    return fig, ax  # Return figure and axis for notebook display

# Convenience function for random effects forest plots
def create_random_effects_forest_plot(random_effects_summary, title, color='navy', n_show=15):
    """
    Create forest plot for random effects (top and bottom effects).
    Uses the main create_forest_plot function under the hood.
    
    Parameters:
    -----------
    random_effects_summary : DataFrame with mean, hdi_3%, hdi_97% columns
    title : str
    color : str
    n_show : int, number of top/bottom effects to show
    """
    if len(random_effects_summary) == 0:
        return
    
    # Sort by effect size and select top/bottom effects
    sorted_effects = random_effects_summary.sort_values('mean')
    n_show = min(n_show, len(sorted_effects))
    
    top_effects = sorted_effects.tail(n_show)
    bottom_effects = sorted_effects.head(n_show)
    combined_effects = pd.concat([bottom_effects, top_effects])
    
    # Calculate OR and CI
    combined_effects = combined_effects.copy()
    combined_effects['OR'] = np.exp(combined_effects['mean'])
    combined_effects['OR_low'] = np.exp(combined_effects['hdi_3%'])
    combined_effects['OR_high'] = np.exp(combined_effects['hdi_97%'])
    
    # Add category information for grouping
    combined_effects['category'] = ['Lowest Effects'] * len(bottom_effects) + ['Highest Effects'] * len(top_effects)
    
    # Use the main forest plot function
    return create_forest_plot(combined_effects, f'{title}\n(Top/Bottom {n_show} Effects)', color)

# Shared function to clean variable names for both plots and tables
def clean_variable_name(var_name, for_plot=False):
    """
    Clean a single variable name for publication-ready display.
    
    Parameters:
    -----------
    var_name : str, original variable name
    for_plot : bool, if True, return format suitable for plots; if False, for tables
    
    Returns:
    --------
    str, cleaned variable name
    """
    clean_name = var_name
    
    # Handle categorical variables with Treatment specification
    if 'C(' in clean_name and 'Treatment(' in clean_name:
        # Extract variable name and category
        parts = clean_name.split("', Treatment('")
        if len(parts) == 2:
            var_part = parts[0].replace("C(", "").replace("'", "")
            ref_part = parts[1].replace("'))", "").replace("[", ": ").replace("]", "")
            
            # Clean variable names
            if var_part == 'PhD_Postdoc':
                var_display = 'PhD/Postdoc'
            elif var_part == 'journal_category':
                var_display = 'Journal Category'
            elif var_part == 'ranking_category':
                var_display = 'University Ranking'
            elif var_part == 'First_Author_Sex':
                var_display = 'First Author Sex'
            elif var_part == 'Leading_Author_Sex':
                var_display = 'Leading Author Sex'
            elif var_part == 'Junior_Senior':
                var_display = 'Seniority'
            elif var_part == 'highest_impact_journal':
                var_display = 'Highest Impact Journal'
            elif var_part == 'highest_ranking_institution':
                var_display = 'Highest Ranking Institution'
            else:
                var_display = var_part.replace('_', ' ').title()
            
            # Get reference category name - clean it up
            ref_clean = ref_part.replace("'", "")
            
            if for_plot:
                clean_name = f"{var_display}: {ref_part} (vs {ref_clean})"
            else:
                clean_name = f"{var_display}: {ref_part}"
    
    # Handle continuous variables
    elif 'log_num_papers' in clean_name:
        clean_name = 'Log(Number of Papers)'
    elif 'year_s' in clean_name:
        clean_name = clean_name.replace('year_s', 'Year Spline ')
    elif 'Intercept' in clean_name:
        clean_name = 'Intercept'
    
    # Handle random effects with author IDs
    elif '[' in clean_name and ']' in clean_name:
        author_id = clean_name.split('[')[1].split(']')[0]
        clean_name = f'Author {author_id[:20]}'  # Truncate long IDs
    
    return clean_name

# Function to clean variable names for tables
def clean_variable_names_for_table(df):
    """
    Clean variable names in DataFrame index for publication-ready tables.
    
    Parameters:
    -----------
    df : DataFrame with variable names as index
    
    Returns:
    --------
    DataFrame with cleaned index names
    """
    df_clean = df.copy()
    clean_names = [clean_variable_name(idx, for_plot=False) for idx in df.index]
    df_clean.index = clean_names
    return df_clean

# Function to format DataFrame for publication (no scientific notation, 2 decimal places)
def format_results_table(df, or_columns=['OR', 'OR_low', 'OR_high'], other_columns=None):
    """
    Format results table for publication with proper decimal formatting.
    
    Parameters:
    -----------
    df : DataFrame with results
    or_columns : list of OR-related columns to format to 2 decimal places
    other_columns : list of other columns to format
    
    Returns:
    --------
    DataFrame with formatted values
    """
    df_formatted = df.copy()
    
    # Format OR columns to 2 decimal places
    for col in or_columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}")
    
    # Format other numeric columns if specified
    if other_columns:
        for col in other_columns:
            if col in df_formatted.columns:
                if 'p' in col.lower() or 'pval' in col.lower():
                    # Special formatting for p-values
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: f"{x:.3f}" if x >= 0.001 else f"{x:.2e}"
                    )
                else:
                    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}")
    
    return df_formatted


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
