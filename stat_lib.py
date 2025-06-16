# Core statistical libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm

# Statistical modeling
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Bayesian analysis
import bambi as bmb
import arviz as az

# Plotting
import matplotlib.pyplot as plt
import forestplot as fp
from plot_info import SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE


def parse_model_variables(variable_names, formula=None):
    """
    Bulletproof function to parse model variables and create clean category mappings.
    
    Parameters:
    -----------
    variable_names : list of str
        List of model variable names (e.g., from model.summary())
    formula : str, optional
        Model formula string for better parsing context
        
    Returns:
    --------
    dict: {
        'clean_names': list of clean variable names for display,
        'categories': list of category names for each variable,
        'category_mapping': dict mapping categories to variable lists,
        'is_categorical': list of booleans indicating categorical variables
    }
    """
    

    clean_names = []
    categories = []
    category_mapping = {}
    is_categorical = []
    
    for var_name in variable_names:
        # Skip intercept
        if 'Intercept' in var_name:
            continue
            
        # Parse categorical variables: C(variable, Treatment(...))[level]
        if var_name.startswith('C('):
            # Extract base variable name
            end_pos = var_name.find(',')
            if end_pos > 0:
                base_var = var_name[2:end_pos].strip()
                
                # Extract level from [level]
                level_start = var_name.rfind('[')
                level_end = var_name.rfind(']')
                if level_start > 0 and level_end > level_start:
                    level = var_name[level_start+1:level_end]
                    # Remove T. prefix if present
                    if level.startswith('T.'):
                        level = level[2:]
                else:
                    level = "Unknown Level"
                
                # Create clean names
                category_name = base_var.replace('_', ' ').title()
                clean_name = level
                
                clean_names.append(clean_name)
                categories.append(category_name)
                is_categorical.append(True)
                
                # Add to category mapping
                if category_name not in category_mapping:
                    category_mapping[category_name] = []
                category_mapping[category_name].append(var_name)
        
        # Parse year splines
        elif var_name.startswith('year_s'):
            spline_num = var_name.replace('year_s', '')
            category_name = 'Year (Splines)'
            clean_name = f"Spline {spline_num}"
            
            clean_names.append(clean_name)
            categories.append(category_name)
            is_categorical.append(False)
            
            if category_name not in category_mapping:
                category_mapping[category_name] = []
            category_mapping[category_name].append(var_name)
        
        # Parse continuous variables
        else:
            category_name = var_name.replace('_', ' ').title()
            clean_name = category_name
            
            clean_names.append(clean_name)
            categories.append(category_name)
            is_categorical.append(False)
            
            if category_name not in category_mapping:
                category_mapping[category_name] = []
            category_mapping[category_name].append(var_name)
    
    return {
        'clean_names': clean_names,
        'categories': categories, 
        'category_mapping': category_mapping,
        'is_categorical': is_categorical
    }

# ============================================================================
# DESCRIPTIVE STATISTICS AND DATA EXPLORATION
# ============================================================================

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

# ============================================================================
# STATISTICAL REPORTING FUNCTIONS
# ============================================================================

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

# ============================================================================
# MODEL UTILITIES AND DIAGNOSTICS  
# ============================================================================

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

def check_model_convergence(short=False, idata=None):
    if idata is None:
        raise ValueError("No inference data provided. Please provide an ArviZ InferenceData object.")

    # Check convergence diagnostics
    summary_stats = az.summary(idata, round_to=2)  # R-hat, ESS

    print(f"min ESS: {summary_stats['ess_bulk'].min():.1f}, max ESS: {summary_stats['ess_bulk'].max():.1f}")
    print(f"min R-hat: {summary_stats['r_hat'].min():.3f}, max R-hat: {summary_stats['r_hat'].max():.3f}")

    if summary_stats['ess_bulk'].min() < 500:
        print("⚠️ ‼️ Warning: ESS is below 2000, indicating potential convergence issues.")
    if summary_stats['r_hat'].max() > 1.01:
        print("⚠️ ‼️ Warning: R-hat is above 1.01, indicating potential convergence issues.")

    loo = az.loo(idata, pointwise=True)
    print(loo)

    if not short:
        az.plot_trace(idata, figsize=(8, 20))
        print(summary_stats)
    return summary_stats

# ============================================================================
# VISUALIZATION AND PLOTTING FUNCTIONS
# ============================================================================

def create_forest_plot(data, title, color='navy'):
    """
    Create publication-friendly forest plots with categorical organization.
    
    Parameters:
    -----------
    data : DataFrame with columns OR, OR_low, OR_high
    title : str, plot title
    color : str, color for points and lines
    
    Returns:
    --------
    tuple of (fig, ax) objects
    """
    if len(data) == 0:
        print("No data to plot")
        return None, None
    
    # Use variable parsing to organize by categories
    var_info = parse_model_variables(data.index.tolist())
    
    # Build plot items with category headers
    plot_items = []
    for category, var_names in var_info['category_mapping'].items():
        # Add category header
        plot_items.append((category, None, True))
        
        # Add variables under this category
        for var_name in var_names:
            if var_name in data.index:
                var_idx = data.index.tolist().index(var_name)
                clean_name = var_info['clean_names'][var_idx]
                indented_name = f"    {clean_name}"  # Indent subcategories
                plot_items.append((indented_name, data.loc[var_name], False))
    
    # Extract components for plotting
    clean_labels = [item[0] for item in plot_items]
    data_for_plot = [item[1] for item in plot_items]
    is_header = [item[2] for item in plot_items]
    y_positions = list(range(len(plot_items)))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(clean_labels)*0.3)))
    
    # Determine plot limits
    all_or_lows = [row['OR_low'] for row in data_for_plot if row is not None]
    all_or_highs = [row['OR_high'] for row in data_for_plot if row is not None]
    
    min_or = min(all_or_lows) if all_or_lows else 0.1
    max_or = max(all_or_highs) if all_or_highs else 10
    
    # Set plot limits with padding
    plot_min = max(min_or * 0.8, 0.01)
    plot_max = min(max_or * 1.2, 100)
    
    # Plot error bars and points 
    for y_pos, data_row in zip(y_positions, data_for_plot):
        if data_row is not None:  # Skip category headers
            or_val = data_row['OR']
            or_low = data_row['OR_low'] 
            or_high = data_row['OR_high']
            
            ax.errorbar(or_val, y_pos,
                        xerr=[[or_val-or_low], [or_high-or_val]],
                        fmt='o', color=color, ecolor='lightgray', capsize=4, 
                        markersize=6, linewidth=2)
    
    # Reference line at OR = 1
    ax.axvline(1, ls='--', color='red', alpha=0.7, linewidth=1)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, fontsize=MEDIUM_SIZE)
    
    # Make category headers bold
    for i, is_hdr in enumerate(is_header):
        if is_hdr:
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_fontsize(BIGGER_SIZE)
    
    ax.set_xlabel("Odds Ratio", fontsize=MEDIUM_SIZE)
    ax.set_xscale('log')
    ax.set_title(title, fontsize=BIGGER_SIZE, pad=20)
    ax.set_xlim(plot_min, plot_max)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig, ax

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
    
    # Use the main forest plot function
    return create_forest_plot(combined_effects, f'{title}\n(Top/Bottom {n_show} Effects)', color)

def create_elegant_forest_plot(data, title, figsize=(10, 8), **kwargs):
    """
    Create an elegant forest plot using the forestplot package.
    
    Parameters:
    -----------
    data : DataFrame with columns OR, OR_low, OR_high and index as variable names
    title : str, plot title
    figsize : tuple, figure size
    **kwargs : additional arguments passed to forestplot
    
    Returns:
    --------
    matplotlib axes object
    """
    # Use bulletproof parsing for variable labels and groups
    var_info = parse_model_variables(data.index.tolist())
    
    # Prepare data for forestplot package
    plot_data = data.copy()
    
    # Use parsed clean names and categories
    plot_data['varlabel'] = var_info['clean_names']
    plot_data['group'] = var_info['categories']
    
    # Add formatted confidence intervals
    plot_data['est_ci'] = plot_data.apply(
        lambda row: f"{row['OR']:.2f} ({row['OR_low']:.2f}, {row['OR_high']:.2f})", 
        axis=1
    )
    
    # Get unique groups for ordering (preserve order from category_mapping)
    unique_groups = list(var_info['category_mapping'].keys())
    
    # Generate appropriate x-ticks for log scale
    min_val = plot_data[['OR_low']].min().min()
    max_val = plot_data[['OR_high']].max().max()
    
    # Create log-spaced ticks
    if min_val < 0.1:
        xticks = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    elif min_val < 0.5:
        xticks = [0.1, 0.5, 1, 2, 5, 10]
    else:
        xticks = [0.5, 1, 2, 5, 10]
    
    # Filter ticks based on data range
    xticks = [x for x in xticks if min_val * 0.5 <= x <= max_val * 2]
    
    # Set up default parameters
    default_params = {
        'estimate': 'OR',
        'll': 'OR_low', 
        'hl': 'OR_high',
        'varlabel': 'varlabel',
        'annote': ['est_ci'],
        'annoteheaders': ['OR (95% CI)'],
        'groupvar': 'group',
        'group_order': unique_groups,
        'xlabel': 'Odds Ratio',
        'sort': False,  # Keep original order
        'table': True,
        'figsize': figsize,
        'logscale': True,
        #'xticks': xticks,
        # Styling parameters
        'marker': 'D',  # Diamond markers
        'markersize': 35,
        'xline': 1,  # Reference line at OR = 1
        'xlinestyle': '--',  # Dashed reference line
        'xlinecolor': '#808080',  # Gray reference line
        'xtick_size': MEDIUM_SIZE,
        'ytick_size': MEDIUM_SIZE,
        'xlabel_size': MEDIUM_SIZE,
        'title_size': BIGGER_SIZE,
        # Table font sizes
        'annote_size': MEDIUM_SIZE,  # Font size for annotations (table content)
        'annoteheaders_size': MEDIUM_SIZE,  # Font size for annotation headers
        'varlabel_size': MEDIUM_SIZE,  # Font size for variable labels
    }
    
    # Update with user-provided kwargs
    default_params.update(kwargs)
    
    # Create the plot
    ax = fp.forestplot(
        plot_data,
        **default_params
    )
    
    # Add title
    ax.set_title(title, fontsize=BIGGER_SIZE, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    return ax

# ============================================================================
# DATA FORMATTING AND EXPORT UTILITIES
# ============================================================================

def format_results_table(df, or_columns=['OR', 'OR_low', 'OR_high'], other_columns=None, clean_variable_names=True):
    """
    Format results table for publication with proper decimal formatting and clean variable names.
    
    Parameters:
    -----------
    df : DataFrame with results
    or_columns : list of OR-related columns to format to 2 decimal places
    other_columns : list of other columns to format
    clean_variable_names : bool, whether to clean variable names in index
    
    Returns:
    --------
    DataFrame with formatted values and cleaned variable names
    """
    df_formatted = df.copy()
    
    # Clean variable names in index if requested
    if clean_variable_names and hasattr(df_formatted.index, 'tolist'):
        var_info = parse_model_variables(df_formatted.index.tolist())
        df_formatted.index = var_info['clean_names']
    
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


