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
            # Extract base variable name and treatment reference
            end_pos = var_name.find(',')
            if end_pos > 0:
                base_var = var_name[2:end_pos].strip()
                
                # Extract treatment reference
                treatment_start = var_name.find('Treatment(')
                treatment_reference = "Unknown"
                if treatment_start > 0:
                    treatment_start += len('Treatment(')
                    treatment_end = var_name.find(')', treatment_start)
                    if treatment_end > treatment_start:
                        treatment_reference = var_name[treatment_start:treatment_end].strip()
                        # Remove quotes if present
                        if treatment_reference.startswith(("'", '"')) and treatment_reference.endswith(("'", '"')):
                            treatment_reference = treatment_reference[1:-1]
                
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
                clean_name = f"{level} (vs {treatment_reference})"
                
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
        
        # Parse random effects: 1|something[value]
        elif '|' in var_name and '[' in var_name and ']' in var_name:
            # Extract value from pattern like "1|something[value]"
            start = var_name.rfind('[')
            end = var_name.rfind(']')
            if start > 0 and end > start:
                clean_name = var_name[start+1:end]
            else:
                clean_name = var_name
            
            # Extract category from the "something" part between | and [
            pipe_pos = var_name.find('|')
            bracket_pos = var_name.find('[')
            if pipe_pos > 0 and bracket_pos > pipe_pos:
                something = var_name[pipe_pos+1:bracket_pos]
                category_name = something.replace('_', ' ').title() + ' Effects'
            else:
                category_name = 'Random Effects'
            
            clean_names.append(clean_name)
            categories.append(category_name)
            is_categorical.append(False)
            
            if category_name not in category_mapping:
                category_mapping[category_name] = []
            category_mapping[category_name].append(var_name)
        
        # Parse specific variables with custom names
        elif var_name == 'num_papers_std':
            category_name = 'Productivity'
            clean_name = 'N Publications'
            
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
    
    # Create mapping from original variable names to clean info
    var_name_mapping = {}
    clean_idx = 0
    for var_name in variable_names:
        if 'Intercept' in var_name:
            var_name_mapping[var_name] = {
                'clean_name': 'Intercept',
                'category': 'Model',
                'is_categorical': False
            }
        else:
            var_name_mapping[var_name] = {
                'clean_name': clean_names[clean_idx],
                'category': categories[clean_idx],
                'is_categorical': is_categorical[clean_idx]
            }
            clean_idx += 1
    
    return {
        'clean_names': clean_names,
        'categories': categories, 
        'category_mapping': category_mapping,
        'is_categorical': is_categorical,
        'var_name_mapping': var_name_mapping  # NEW: direct mapping
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

def generate_forest_plot_ticks(data, or_columns=['OR', 'OR_low', 'OR_high']):
    """
    Generate appropriate x-axis ticks and limits for forest plots based on data range.
    Ensures at least one tick on each side of 1 (reference line) and that no 
    confidence intervals ≥ 0.1 are cut off.
    
    Parameters:
    -----------
    data : DataFrame with OR columns
    or_columns : list of column names containing OR values
    
    Returns:
    --------
    tuple: (ticks_list, plot_min, plot_max)
    """
    # Get all OR values from the specified columns
    all_values = []
    for col in or_columns:
        if col in data.columns:
            all_values.extend(data[col].dropna().tolist())
    
    if not all_values:
        return [0.1, 0.5, 1, 2, 5, 10], 0.05, 20  # Default ticks and limits
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Determine tick spacing based on data range
    data_range = max_val / min_val
    
    if data_range > 100:  # Large range, use x10 spacing
        base_ticks = [0.01, 0.1, 1, 10, 100, 1000]
    else:  # Smaller range, use x5 spacing  
        base_ticks = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    
    # Filter ticks to reasonable range around data
    # Extend beyond data by factor of 1.5 on each side (less aggressive)
    tick_min = min_val / 1.5
    tick_max = max_val * 1.5
    
    filtered_ticks = [tick for tick in base_ticks if tick_min <= tick <= tick_max]
    
    # Ensure 1 is always included
    if 1 not in filtered_ticks:
        filtered_ticks.append(1)
    
    # Ensure we have at least one tick on each side of 1, but don't force symmetry
    if not any(tick < 1 for tick in filtered_ticks):
        # Add the largest tick below 1 that's reasonable for the data
        candidates_below = [tick for tick in base_ticks if tick < 1 and tick >= min_val / 3]
        if candidates_below:
            filtered_ticks.append(max(candidates_below))
        else:
            filtered_ticks.append(0.1)  # fallback
    
    if not any(tick > 1 for tick in filtered_ticks):
        # Add the smallest tick above 1 that's reasonable for the data
        candidates_above = [tick for tick in base_ticks if tick > 1 and tick <= max_val * 3]
        if candidates_above:
            filtered_ticks.append(min(candidates_above))
        else:
            filtered_ticks.append(10)  # fallback
    
    ticks = sorted(list(set(filtered_ticks)))
    
    # Calculate plot limits ensuring nothing ≥ 0.1 is cut
    tick_based_min = min(ticks) * 0.8
    tick_based_max = max(ticks) * 1.2
    
    # Ensure data limits are respected, especially for values ≥ 0.1
    plot_min = min(tick_based_min, min_val * 0.9) if min_val >= 0.1 else tick_based_min
    plot_max = max(tick_based_max, max_val * 1.1)
    
    return ticks, plot_min, plot_max

def create_forest_plot(data, title="", color='black'):
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
    original_var_list = data.index.tolist()
    var_info = parse_model_variables(original_var_list)
    
    # Build plot items with category headers (reversed order for proper display)
    plot_items = []
    categories_list = list(var_info['category_mapping'].items())
    categories_list.reverse()  # Reverse to show categories above their items
    
    for category, var_names in categories_list:
        # Add variables under this category first
        var_items = []
        for var_name in var_names:
            if var_name in data.index:
                clean_name = var_info['var_name_mapping'][var_name]['clean_name']
                indented_name = clean_name  # No text indentation, handled by positioning
                var_items.append((indented_name, data.loc[var_name], False))
        
        # Add variables in reverse order
        var_items.reverse()
        plot_items.extend(var_items)
        
        # Add category header after variables (will appear above due to y-axis direction)
        plot_items.append((category, None, True))
    
    # Extract components for plotting
    clean_labels = [item[0] for item in plot_items]
    data_for_plot = [item[1] for item in plot_items]
    is_header = [item[2] for item in plot_items]
    y_positions = list(range(len(plot_items)))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, max(4, len(clean_labels)*0.6)))
    
    # Generate appropriate ticks and plot limits
    ticks, plot_min, plot_max = generate_forest_plot_ticks(data)
    
    # Plot error bars and points 
    for y_pos, data_row in zip(y_positions, data_for_plot):
        if data_row is not None:  # Skip category headers
            or_val = data_row['OR']
            or_low = data_row['OR_low'] 
            or_high = data_row['OR_high']
            
            ax.errorbar(or_val, y_pos,
                        xerr=[[or_val-or_low], [or_high-or_val]],
                        fmt='o', color=color, ecolor='black', capsize=4, 
                        markersize=6, linewidth=2)
    
    # Reference line at OR = 1
    ax.axvline(1, ls='--', color='red', alpha=0.7, linewidth=1)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, fontsize=MEDIUM_SIZE)
    
    # Position category headers and items completely outside plot area
    for i, is_hdr in enumerate(is_header):
        if is_hdr:
            # Category headers: bold, left-aligned, outside plot area
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_fontsize(MEDIUM_SIZE)
            ax.get_yticklabels()[i].set_horizontalalignment('left')
            ax.get_yticklabels()[i].set_x(-0.35)  # Far left, outside plot
        else:
            # Category items: normal weight, indented but outside plot
            ax.get_yticklabels()[i].set_horizontalalignment('left')
            ax.get_yticklabels()[i].set_x(-0.30)  # Indented from category headers
    
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=MEDIUM_SIZE)
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_title(title, fontsize=BIGGER_SIZE, pad=20)
    ax.set_xlim(plot_min, plot_max)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label overlap with plot area
    plt.tight_layout()
    #plt.subplots_adjust(left=0.1)  # Increase left margin more for labels
    return fig, ax

def create_forest_plot2(data, title="", color='darkblue'):
    """
    Create forest plot with custom categories and labels, working with any data format.
    Based on create_forest_plot but with custom organization and inline OR values.
    
    Parameters:
    -----------
    data : DataFrame with OR, OR_low, OR_high columns and variable names as index
    title : str
    color : str
    
    Returns:
    --------
    tuple of (fig, ax) objects
    """
    if len(data) == 0:
        print("No data to plot")
        return None, None
    
    # Define custom categories with patterns for both cleaned and raw variable names
    custom_categories = {
        'Journals Impact Factor': [
            ('Journal Category: High Impact', 'High vs Low Impact'),
            ('Journal Category: Trophy Journals', 'Trophy vs Low Impact'),
            # Raw patterns
            ("journal_category, Treatment('Low Impact'))[High Impact]", 'High vs Low Impact'),
            ("journal_category, Treatment('Low Impact'))[Trophy Journals]", 'Trophy vs Low Impact')
        ],
        'Shangai University Rank': [
            ('Ranking Category: Top 50', 'Top 50 vs Not Ranked'),
            ('Ranking Category: 51-100', '51-100 vs Not Ranked'), 
            ('Ranking Category: 101+', '101+ vs Not Ranked'),
            # Raw patterns
            ("ranking_category, Treatment('Not Ranked'))[Top 50]", 'Top 50 vs Not Ranked'),
            ("ranking_category, Treatment('Not Ranked'))[51-100]", '51-100 vs Not Ranked'),
            ("ranking_category, Treatment('Not Ranked'))[101+]", '101+ vs Not Ranked')
        ],
        'Date of publication': [
            ('Year (Splines): Spline 1', 'Spline 1'),
            ('Year (Splines): Spline 2', 'Spline 2'),
            ('Year (Splines): Spline 3', 'Spline 3'),
            # Raw patterns
            ('year_s1', 'Spline 1'),
            ('year_s2', 'Spline 2'),
            ('year_s3', 'Spline 3')
        ],
        'First Authors Variables': [
            ('First Author Sex: Female', 'Female vs Male'),
            ('Phd Postdoc: Post-doc', 'Post-doc vs PhD'),
            ('PhD_Postdoc: Post-doc', 'Post-doc vs PhD'),
            # Raw patterns
            ("First_Author_Sex, Treatment('Male'))[Female]", 'Female vs Male'),
            ("PhD_Postdoc, Treatment('PhD'))[Post-doc]", 'Post-doc vs PhD')
        ],
        'Leading Author Variables': [
            ('Leading Author Sex: Female', 'Female vs Male'),
            ('Junior Senior: Junior PI', 'Junior vs Senior'),
            ('F And L: True', 'F and L (after 1995)'),
            ('F And L: Origininal F and L', 'F and L (original vs exploratory)'),
            ('First Paper Before 1995: True', 'First paper before 1995 (Yes vs No)'),
            ('Continuity: True', 'Continuity vs Exploratory'),
            # Raw patterns
            ("Leading_Author_Sex, Treatment('Male'))[Female]", 'Female vs Male'),
            ("Junior_Senior, Treatment('Senior PI'))[Junior PI]", 'Junior vs Senior'),
            ("F_and_L, Treatment('False'))[True]", 'First Author in the field\n(Yes vs No)'),
            #("F_and_L, Treatment('False'))[Origininal F and L]", 'F and L (original vs exploratory)'),
            ('first_paper_before_1995, Treatment(False))[True]', 'First paper before 1995 \n(Yes vs No)'),
            ('Continuity, Treatment(False))[True]', 'Continuity vs Exploratory')
        ]
    }
    
    # Handle case where data doesn't have variable names as index
    if 'Unnamed: 0' in data.columns:
        # Use the 'Unnamed: 0' column as variable names
        var_names = data['Unnamed: 0'].tolist()
        data_dict = data.set_index('Unnamed: 0').to_dict('index')
    else:
        # Use the index as variable names
        var_names = data.index.tolist()
        data_dict = data.to_dict('index')
    
    # Organize variables by custom categories using simple pattern matching
    organized_categories = {}
    
    for var_name in var_names:
        # Skip if var_name is not a string
        if not isinstance(var_name, str):
            continue
            
        for category, patterns in custom_categories.items():
            for pattern, custom_label in patterns:
                if pattern in var_name:
                    if category not in organized_categories:
                        organized_categories[category] = []
                    
                    # Get data row
                    data_row = data_dict[var_name] if var_name in data_dict else data.loc[var_name]
                    
                    # Store just the custom label for now, we'll add OR values later
                    display_label = custom_label
                    
                    organized_categories[category].append((display_label, data_row, False))
                    break
    
    # Build plot items in desired order
    plot_items = []
    #category_order = ['Last Authors', 'First Authors', 'Years', 'University Ranking', 'Journals']
    category_order = reversed([k for k,v in custom_categories.items()])
    for category in category_order:
        if category in organized_categories:
            var_items = organized_categories[category]
            
            # Add variables in reverse order for proper display
            var_items.reverse()
            plot_items.extend(var_items)
            
            # Add category header
            plot_items.append((category, None, True))
    
    if not plot_items:
        print("No matching variables found for custom categories")
        return None, None
    
    # Extract components for plotting
    clean_labels = [item[0] for item in plot_items]
    data_for_plot = [item[1] for item in plot_items]
    is_header = [item[2] for item in plot_items]
    y_positions = list(range(len(plot_items)))
    
    # Create the plot with wider figure for better OR alignment
    fig, ax = plt.subplots(figsize=(22, max(6, len(clean_labels)*0.65)))
    
    # Generate appropriate ticks and plot limits
    ticks, plot_min, plot_max = generate_forest_plot_ticks(data)
    
    # Plot error bars and points 
    for y_pos, data_row in zip(y_positions, data_for_plot):
        if data_row is not None:  # Skip category headers
            or_val = data_row['OR']
            or_low = data_row['OR_low'] 
            or_high = data_row['OR_high']
            
            ax.errorbar(or_val, y_pos,
                        xerr=[[or_val-or_low], [or_high-or_val]],
                        fmt='o', color=color, ecolor='black', capsize=4, 
                        markersize=6, linewidth=2)
    
    # Reference line at OR = 1
    ax.axvline(1, ls='--', color='red', alpha=0.7, linewidth=1)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, fontsize=MEDIUM_SIZE)
    
    # Add header for OR values column
    ax.text(-0.25, len(plot_items)-1, "Odds Ratio (94% HDI)", transform=ax.get_yaxis_transform(), 
           fontsize=BIGGER_SIZE, va='center', ha='left', fontweight='bold')
    
    # Position category headers and items, and add OR values between labels and forest plot
    for i, (is_hdr, data_row) in enumerate(zip(is_header, data_for_plot)):
        if is_hdr:
            # Category headers: bold, left-aligned, far left
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_fontsize(BIGGER_SIZE)
            ax.get_yticklabels()[i].set_horizontalalignment('left')
            ax.get_yticklabels()[i].set_x(-0.8)  # Far left for category headers
        else:
            # Category items: normal weight, positioned with variable names on far left
            ax.get_yticklabels()[i].set_horizontalalignment('left')
            ax.get_yticklabels()[i].set_x(-0.7)  # Position labels further left
            
            # Add OR values as separate text between labels and forest plot
            if data_row is not None:
                or_val = f"{data_row['OR']:.2f}"
                ci_low = f"{data_row['OR_low']:.2f}"
                ci_high = f"{data_row['OR_high']:.2f}"
                or_text = f"{or_val} ({ci_low}, {ci_high})"
                
                # Add OR text positioned between labels and forest plot
                ax.text(-0.2, i, or_text, transform=ax.get_yaxis_transform(), 
                       fontsize=MEDIUM_SIZE, va='center', ha='left')
    
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=BIGGER_SIZE)
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_title(title, fontsize=BIGGER_SIZE, pad=20)
    ax.set_xlim(plot_min, plot_max)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    return fig, ax

# Convenience function for random effects forest plots
def create_random_effects_forest_plot(random_effects_summary, title, color='navy', n_show=15):
    """
    Create forest plot for random effects (top and bottom effects).
    Creates a simple forest plot without categorical organization.
    
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
    
    # Create simple forest plot without categories
    fig, ax = plt.subplots(figsize=(12, max(4, len(combined_effects)*0.6)))
    
    # Generate appropriate ticks and plot limits
    ticks, plot_min, plot_max = generate_forest_plot_ticks(combined_effects)
    
    # Use cleaned labels from parse_model_variables
    y_positions = range(len(combined_effects))
    original_var_list = combined_effects.index.tolist()
    var_info = parse_model_variables(original_var_list)
    # Create labels using direct mapping
    labels = [var_info['var_name_mapping'][name]['clean_name'] for name in original_var_list]
    
    # Plot error bars and points 
    for i, (_, row) in enumerate(combined_effects.iterrows()):
        or_val = row['OR']
        or_low = row['OR_low'] 
        or_high = row['OR_high']
        
        ax.errorbar(or_val, i,
                    xerr=[[or_val-or_low], [or_high-or_val]],
                    fmt='o', color=color, ecolor='black', capsize=4, 
                    markersize=6, linewidth=2)
    
    # Reference line at OR = 1
    ax.axvline(1, ls='--', color='red', alpha=0.7, linewidth=1)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=MEDIUM_SIZE)
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=MEDIUM_SIZE)
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_title(f'{title}\n(Top/Bottom {n_show} Effects)', fontsize=BIGGER_SIZE, pad=20)
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

def create_elegant_forest_plot(data, title="", figsize=(15, 8), **kwargs):
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
    original_var_list = data.index.tolist()
    var_info = parse_model_variables(original_var_list)
    
    # Prepare data for forestplot package
    plot_data = data.copy()
    
    # Create proper labels and groups using direct mapping
    labels = [var_info['var_name_mapping'][name]['clean_name'] for name in original_var_list]
    groups = [var_info['var_name_mapping'][name]['category'] for name in original_var_list]
    
    plot_data['varlabel'] = labels
    plot_data['group'] = groups
    

    # Add formatted confidence intervals
    plot_data['est_ci'] = plot_data.apply(
        lambda row: f"{row['OR']:.2f} ({row['OR_low']:.2f}, {row['OR_high']:.2f})", 
        axis=1
    )
    
    # Reset index to ensure clean data for forestplot package
    plot_data = plot_data.reset_index(drop=True)
    
    # Get unique groups for ordering - use actual groups in the data to avoid misalignment
    unique_groups = plot_data['group'].unique().tolist()
    
    # Generate appropriate x-ticks using our custom function
    xticks, _, _ = generate_forest_plot_ticks(plot_data)
    print(f"Generated x-ticks: {xticks}")
    
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
        'xlabel': 'Odds Ratio (log scale)',
        'sort': False,  # Keep original order
        'table': True,
        'figsize': figsize,
        'logscale': True,
        'xticks': xticks,
        # Styling parameters
        'marker': 'D',  # Diamond markers
        'markersize': 35,
        'xline': 1,  # Reference line at OR = 1
        'xlinestyle': '--',  # Dashed reference line
        'xlinecolor': '#000000',  # Gray reference line
        'xtick_size': MEDIUM_SIZE,
        'ytick_size': MEDIUM_SIZE,
        'xlabel_size': SMALL_SIZE,
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
        original_var_list = df_formatted.index.tolist()
        var_info = parse_model_variables(original_var_list)
        # For tables, concatenate category and clean name
        table_names = []
        for orig_name in original_var_list:
            var_mapping = var_info['var_name_mapping'][orig_name]
            clean_name = var_mapping['clean_name']
            category = var_mapping['category']
            if 'Intercept' in orig_name:
                table_names.append(clean_name)
            else:
                table_names.append(f"{category}: {clean_name}")
        df_formatted.index = table_names
    
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


