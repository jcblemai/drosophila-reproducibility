import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
from scipy import stats
# import matplotlib as mpl  # Not needed - using direct imports
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
from matplotlib.ticker import PercentFormatter, MaxNLocator
import seaborn as sns


import matplotlib.gridspec as gridspec

# Custom colors for categories

# Set style parameters
plt.style.use('seaborn-v0_8-white')
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

# Standardized figure sizes for consistency
HORIZONTAL_LAYOUT = (20, 8)    # For side-by-side panels (AB)
COMPLEX_LAYOUT = (15, 17)      # For multi-panel figures (ABC)
VERTICAL_LAYOUT = (9,14)     # For stacked panels
SINGLE_PANEL = (10, 8)         # For individual plots

# Panel label font size (for A, B, C labels)
PANEL_LABEL_SIZE = 28

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

assessment_columns = ['Unchallenged', 'Verified', 'Partially Verified', 'Mixed', 'Challenged']

### NORMAL ASSESSEMENT CATEGORIES ###

# Define colors for assessment categories
ASSESSMENT_COLORS = {
    'Challenged': '#e74c3c',    # Red
    'Mixed': '#f39c12',         # Orange (changed from gray)
    'Unchallenged': '#95a5a6',  # Gray (changed from blue)
    'Partially Verified': '#e9f241',  # Yellow
    'Verified': '#2ecc71'       # Green
}
ASSESSMENT_COLORS = {
    'Challenged': '#c0392b',          # Deep red
    'Mixed': '#d68910',               # Warm amber  
    'Unchallenged': '#85929e',        # Neutral gray
    'Partially Verified': '#f1c40f',  # Golden yellow
    'Verified': '#27ae60'             # Forest green
}
# Define the assessment category order for consistent plotting (reversed)
                    # TOP                                                          # Bottom
ASSESSMENT_ORDER = ['Challenged', 'Mixed', 'Partially Verified', 'Unchallenged', 'Verified']

def group_assessment(assessment):
    """Group assessment types into broader categories."""
    if pd.isna(assessment) or assessment == 'Not assessed':
        print(f"Warning: {assessment} is unclear")
        return None
    if 'Partially verified' in str(assessment):
        return 'Partially Verified'
    elif 'Verified' in str(assessment):
        return 'Verified'
    elif 'Unchallenged' in str(assessment):
        return 'Unchallenged'
    elif 'Challenged' in str(assessment):
        return 'Challenged'
    elif 'Mixed' in str(assessment):
        return 'Mixed'

    else:
        print(f"Warning: {assessment} is unclear")
        return None

### EXPANDED ASSESSEMENT CATEGORIES ###

ASSESSMENT_COLORS_EXPANDED = {
    'Challenged in literature': '#e74c3c',                                # Main red
    'Challenged by reproducibility project': '#c0392b',     # Darker red
    
    'Mixed': '#f39c12',                                     # Orange
    
    'Unchallenged': '#95a5a6',                              # Main gray
    'Unchallenged, logically consistent': '#bdc3c7',        # Light gray
    'Unchallenged, logically inconsistent': '#7f8c8d',      # Dark gray
    
    'Partially Verified': '#e9f241',                        # Yellow
    'Verified': '#2ecc71'                                   # Green
}

ASSESSMENT_ORDER_EXPANDED = [
    'Challenged by reproducibility project',
    'Challenged in literature',
    'Mixed',
    'Unchallenged, logically inconsistent',
    'Unchallenged',
    'Unchallenged, logically consistent',
    'Partially Verified',
    'Verified'
]

# Create a mapping from detailed to standard categories for label aggregation
# TODO This mapping is not waht shoudl be used for the table, genearlly
category_mapping = {
    'Challenged by reproducibility project': 'Challenged',
    'Challenged in literature': 'Challenged',
    'Mixed': 'Mixed',
    'Unchallenged, logically inconsistent': 'Unchallenged',
    'Unchallenged': 'Unchallenged',
    'Unchallenged, logically consistent': 'Unchallenged',
    'Partially Verified': 'Partially Verified',
    'Verified': 'Verified'
}

    
def group_detailed_assessment(assessment):
    """Group assessment types into detailed categories."""
    if pd.isna(assessment) or assessment == 'Not assessed':
        print(f"Warning: {assessment} is unclear")
        return None
    if assessment == 'Verified' or assessment == "Verified by same authors" or assessment == "Verified by reproducibility project":
        return 'Verified'
    elif assessment == 'Challenged by reproducibility project':
        return 'Challenged by reproducibility project'
    elif assessment == 'Challenged':
        return 'Challenged in literature'
    elif assessment == 'Mixed':
        return 'Mixed'
    elif assessment == 'Partially verified':
        return 'Partially Verified'
    elif assessment == 'Unchallenged, logically consistent':
        return 'Unchallenged, logically consistent'
    elif assessment == 'Unchallenged, logically inconsistent':
        return 'Unchallenged, logically inconsistent'
    elif assessment == 'Unchallenged':
        return 'Unchallenged'
    else:
        print(f"Warning: {assessment} is unclear")
        return None


# Use global detailed mapping but with updated labels
sankey_detailed_mapping = {
    'Verified': {
        'Verified in literature': ['Verified'],
        'Verified by reproducibility project': ['Verified by reproducibility project'],
        'Verified by same authors': ['Verified by same authors']
    },
    'Challenged': {
        'Challenged in literature': ['Challenged'],
        'Challenged by reproducibility project': ['Challenged by reproducibility project'],
        'Challenged by same authors': ['Challenged by same authors']
    },
    'Unchallenged': {
        'Logically consistent': ['Unchallenged, logically consistent'],
        'Logically inconsistent': ['Unchallenged, logically inconsistent'],
        'General unchallenged': ['Unchallenged'],
        'Selected for manual reproduction': ['Verified by reproducibility project', 'Challenged by reproducibility project']
    }
}

def categorize_journal(impact_factor):
    """Categorize journals based on impact factor."""
    if pd.isna(impact_factor):
        return None
    if impact_factor >= 50:
        return 'Trophy Journals'
    elif impact_factor >= 10:
        return 'High Impact'
    else:
        return 'Low Impact'

def bin_years(year):
    """Bin years into predefined time periods."""
    if pd.isna(year):
        return None
    if year <= 1991:
        return '≤1991'
    elif year <= 1996:
        return '1992-1996'
    elif year <= 2001:
        return '1997-2001'
    elif year <= 2006:
        return '2002-2006'
    else:
        return '2007-2011'

def hex_to_rgba(hex_color, alpha=0.5):
    """Convert hex color to rgba with alpha."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def adjust_color(hex_color, factor=0.8):
    """Lighten or darken a hex color by a factor."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Lighten the color
    r = min(255, int(r + (255 - r) * (1 - factor)))
    g = min(255, int(g + (255 - g) * (1 - factor)))
    b = min(255, int(b + (255 - b) * (1 - factor)))
    
    return f'#{r:02x}{g:02x}{b:02x}'

def create_stacked_bar_plot(df, mode='absolute', by_time=False, use_expanded=False, ax=None):
    """
    Create a stacked bar plot of major claims.
    
    Parameters:
    - df: DataFrame with claims data
    - mode: 'absolute' for counts or 'percentage' for percentages
    - by_time: If True, group by time periods; if False, group by journal categories
    - use_expanded: If True, use expanded assessment categories
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Apply categorizations based on specified detail level
    if use_expanded:
        df_copy.loc[:, 'assessment_type_grouped'] = df_copy['assessment_type'].apply(group_detailed_assessment)
        assessment_order = ASSESSMENT_ORDER_EXPANDED
        assessment_colors = ASSESSMENT_COLORS_EXPANDED
        # Category mapping is globally defined for expanded mode
    else:
        df_copy.loc[:, 'assessment_type_grouped'] = df_copy['assessment_type'].apply(group_assessment)
        assessment_order = ASSESSMENT_ORDER
        assessment_colors = ASSESSMENT_COLORS
        category_mapping = {cat: cat for cat in ASSESSMENT_ORDER}  # Identity mapping
    
    if by_time:
        # Group by time periods
        df_copy.loc[:, 'group_by'] = df_copy['year'].apply(bin_years)
        group_order = ['≤1991', '1992-1996', '1997-2001', '2002-2006', '2007-2011']
        x_label = ''
    else:
        # Group by journal categories
        df_copy.loc[:, 'group_by'] = df_copy['impact_factor'].apply(categorize_journal)
        group_order = ['Low Impact', 'High Impact', 'Trophy Journals']
        x_label = 'Journal Category'
    
    # Filter out rows with missing values
    filtered_df = df_copy[df_copy['group_by'].notna() & df_copy['assessment_type_grouped'].notna()].copy()
    print(f"Using {len(filtered_df)} of {len(df_copy)} rows")
    if len(filtered_df) != len(df_copy):
        print("🛑 missing rows:")
        print(df_copy[~(df_copy['group_by'].notna() & df_copy['assessment_type_grouped'].notna())][["assessment_type_grouped", "group_by"]])
    
    # Add standard category column for aggregation
    filtered_df.loc[:, 'standard_category'] = filtered_df['assessment_type_grouped'].map(lambda x: category_mapping.get(x, x))
    
    # Create pivot tables
    detailed_pivot = pd.pivot_table(
        filtered_df,
        values='article_id',
        index='group_by',
        columns='assessment_type_grouped',
        aggfunc='count',
        fill_value=0
    )
    
    # Create standard pivot only if using expanded categories
    if use_expanded:
        standard_pivot = pd.pivot_table(
            filtered_df,
            values='article_id',
            index='group_by',
            columns='standard_category',
            aggfunc='count',
            fill_value=0
        )
    else:
        # In non-expanded mode, detailed and standard are the same
        standard_pivot = detailed_pivot
    
    # Reorder indices 
    detailed_pivot = detailed_pivot.reindex(index=group_order)
    standard_pivot = standard_pivot.reindex(index=group_order)

    # Calculate the total counts for each group
    row_totals = standard_pivot.sum(axis=1)
    
    # Calculate percentages based on row totals
    detailed_pct = detailed_pivot.div(row_totals, axis=0) * 100
    standard_pct = standard_pivot.div(row_totals, axis=0) * 100
    
    # Choose which data to use for plotting
    if mode == 'absolute':
        detailed_plot_data = detailed_pivot
        standard_plot_data = standard_pivot
    else:  # 'percentage' mode
        detailed_plot_data = detailed_pct
        standard_plot_data = standard_pct

    
    # Create / reuse axes ------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Initialize bottom positions for stacking bars
    bottom = np.zeros(len(detailed_plot_data))
    
    # Dictionary to track bottom and height of each standard category
    category_positions = {}
    for cat in set(category_mapping.values()) if use_expanded else assessment_order:
        category_positions[cat] = {
            'bottom': np.zeros(len(detailed_plot_data)),
            'height': np.zeros(len(detailed_plot_data))
        }
    
    # Store handles for legend
    handles = []
    labels = []
    
    # Plot each assessment category in reverse order (for proper stacking)
    for cat in reversed(assessment_order):
        if cat in detailed_plot_data.columns:
            # Get standard category for this detailed category
            std_cat = category_mapping.get(cat, cat)
            
            # Store the starting position for this category
            current_bottom = bottom.copy()
            category_positions[std_cat]['bottom'] = current_bottom.copy()
            
            # Plot the bar
            bar = ax.bar(
                detailed_plot_data.index,
                detailed_plot_data[cat],
                bottom=bottom,
                color=assessment_colors[cat],
                edgecolor="white",
                linewidth=0.5
            )
            
            # Store handle and label for legend
            handles.append(bar)
            labels.append(cat)
            
            # Update bottom for next bar
            bottom += detailed_plot_data[cat]
            
            # Calculate and store total height for this standard category
            category_heights = bottom - current_bottom
            category_positions[std_cat]['height'] += category_heights
    
    # Add labels for standard categories - ONLY in percentage mode
    if mode == 'percentage':  # Only add percentage labels in percentage mode
        standard_categories = set(category_mapping.values()) if use_expanded else assessment_order
        for std_cat in standard_categories:
            if std_cat in standard_plot_data.columns:
                for i, group in enumerate(standard_plot_data.index):
                    # Only add label if there's a non-zero value
                    if standard_plot_data.loc[group, std_cat] > 0:
                        # Calculate position for label
                        if use_expanded:
                            # Find which detailed categories belong to this standard category
                            detailed_cats = [cat for cat in assessment_order if category_mapping.get(cat) == std_cat]
                            
                            # Calculate the bottom position and total height
                            bottom_positions = []
                            total_height = 0
                            
                            for cat in detailed_cats:
                                if cat in detailed_plot_data.columns and detailed_plot_data.loc[group, cat] > 0:
                                    # Find where this category starts in the stack
                                    cat_bottom = 0
                                    for prev_cat in reversed(assessment_order):
                                        if prev_cat == cat:
                                            break
                                        if prev_cat in detailed_plot_data.columns:
                                            cat_bottom += detailed_plot_data.loc[group, prev_cat]
                                    
                                    bottom_positions.append(cat_bottom)
                                    total_height += detailed_plot_data.loc[group, cat]
                            
                            if not bottom_positions:  # Skip if no categories have data
                                continue
                                
                            bottom_pos = min(bottom_positions)
                            height = total_height
                        else:
                            # For standard categories, use the precalculated positions
                            bottom_pos = category_positions[std_cat]['bottom'].iloc[i] if hasattr(category_positions[std_cat]['bottom'], 'iloc') else category_positions[std_cat]['bottom'][i]
                            height = category_positions[std_cat]['height'].iloc[i] if hasattr(category_positions[std_cat]['height'], 'iloc') else category_positions[std_cat]['height'][i]
                        
                        # Skip if height is zero (avoid division by zero)
                        if height <= 0:
                            continue
                        
                        # Calculate center position
                        center_pos = bottom_pos + (height / 2)
                        
                        # Get the percentage for this category (ensure scalar)
                        pct_val = standard_pct.loc[group, std_cat]
                        pct = float(pct_val.item() if hasattr(pct_val, 'item') else pct_val)
                        count_val = standard_pivot.loc[group, std_cat]
                        count = int(count_val.item() if hasattr(count_val, 'item') else count_val)
                        
                        # Skip small categories except for 'Challenged'
                        if pct <= 5 and std_cat != 'Challenged':
                            continue
                            
                        # Format as percentage with consistent logic (pct is already 0-100)
                        label_text = f'{pct:.0f}%' if pct >= 10 else f'{pct:.1f}%'
                        
                        # Add the text label with better formatting
                        ax.text(
                            i, center_pos,
                            label_text,
                            ha='center', va='center',
                            color='white', fontweight='bold',
                            fontsize=SMALL_SIZE
                        ).set_path_effects([withStroke(linewidth=3, foreground='black', alpha=0.7)])

        # Add total counts at the top of each column
        for i, group in enumerate(group_order):
            if group in row_totals.index:
                total = int(row_totals[group])
                # Position the text slightly above the top of the bar
                top_pos = (bottom.iloc[i] if hasattr(bottom, 'iloc') else bottom[i]) + 1  # Add a small offset
                ax.text(
                    i, top_pos,
                    f'$n_c$={total}',
                    ha='center', va='bottom',
                    fontweight='normal',
                    fontsize=MEDIUM_SIZE,
                    usetex=True
                )
    
    # Customize the plot with consistent styling
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set titles and labels
    if mode == 'absolute':
        y_label = 'Number of Claims'
        title_prefix = ''
    else:
        y_label = '% of Claims'
        title_prefix = 'Distribution of '
    
    if by_time:
        title = f'{title_prefix}Major Claims by Time Period and Assessment Type'
    else:
        title = f'{title_prefix}Major Claims by Journal Category and Assessment Type'
    
    if use_expanded:
        title = title.replace('Assessment Type', 'Detailed Assessment Type')
    
    #ax.set_title(title, pad=20)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    
    # Modify x-tick labels to include journal counts for consistency with create_horizontal_bar_chart
    if not by_time and mode == 'percentage':  # Only show for journal category plots in percentage mode
        # Count unique journals in each category
        journal_counts = {}
        for group in group_order:
            if group in filtered_df['group_by'].values:
                group_data = filtered_df[filtered_df['group_by'] == group]
                # Count unique journals by counting unique journal names or impact factors
                if 'journal_name' in df.columns:
                    unique_journals = group_data['journal_name'].nunique()
                else:
                    # Fallback: estimate from impact factor ranges
                    if group == 'Low Impact':
                        unique_journals = len(df[(df['impact_factor'] < 10) & (df['impact_factor'].notna())]['journal_name'].unique()) if 'journal_name' in df.columns else 0
                    elif group == 'High Impact':
                        unique_journals = len(df[(df['impact_factor'] >= 10) & (df['impact_factor'] < 50) & (df['impact_factor'].notna())]['journal_name'].unique()) if 'journal_name' in df.columns else 0
                    elif group == 'Trophy Journals':
                        unique_journals = len(df[(df['impact_factor'] >= 50) & (df['impact_factor'].notna())]['journal_name'].unique()) if 'journal_name' in df.columns else 0
                    else:
                        unique_journals = 0
                journal_counts[group] = unique_journals
        
        # Update x-tick labels to include journal counts
        new_labels = []
        for i, group in enumerate(group_order):
            if group in journal_counts:
                new_labels.append(f'{group}\n({journal_counts[group]})')
            else:
                new_labels.append(group)
        
        # Set both ticks and labels explicitly to avoid warnings
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(new_labels)
    
    # Ensure y-axis has enough room for the total count labels
    y_max = max(bottom) * 1.1  # Add 10% padding
    ax.set_ylim(0, y_max)
    
    # Add percentage formatting to y-axis for consistency
    if mode == 'percentage':
        ax.yaxis.set_major_formatter(PercentFormatter(100.0))  # Data is in 0-100 range, not 0-1
    
    # Rotate x-axis labels if needed
    # if by_time:
    #     plt.xticks(rotation=45)
    
    # Reverse handles and labels to match the original order (not reversed)
    handles = handles[::-1]
    labels = labels[::-1]
    
    # Add legend with corrected order and consistent styling
    legend = ax.legend(
        handles=handles,
        labels=labels,
        title='Assessment Category',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        frameon=True,
        framealpha=0.9,
        edgecolor='lightgray'
    )
    legend.get_title().set_fontweight('bold')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax





def create_sankey_diagram(df):
    """
    Create a Sankey diagram for claim assessment flow.
    
    Parameters:
    - df: DataFrame with claims data
    
    Returns:
    - fig: Plotly figure object
    """
    # Use global color scheme
    base_colors = ASSESSMENT_COLORS
    
    # Define blue color for the reproduction branch
    reproduction_color = '#3498db'  # Nice blue color
    
    # Node label mappings for display
    node_label_mappings = {
        'Selected for manual reproduction': 'Selected for manual reproduction',
        'Verified by reproducibility project': 'Verified',
        'Challenged by reproducibility project': 'Challenged'
    }
    
    # Count claims
    nodes = []
    node_labels = []
    source = []
    target = []
    value = []
    node_colors = []
    link_colors = []  # Initialize link_colors list here
    
    # Add root node
    total_claims = len(df)
    nodes.append('All Major Claims')
    node_labels.append(f'All Major Claims ({total_claims})')
    node_colors.append('#2c3e50')
    
    # Calculate counts for reproducibility project items
    verified_repro_count = len(df[df['assessment_type'] == 'Verified by reproducibility project'])
    challenged_repro_count = len(df[df['assessment_type'] == 'Challenged by reproducibility project'])
    repro_total = verified_repro_count + challenged_repro_count
    
    # First level: main categories with adjusted counts
    first_level_counts = {}
    for category in ['Verified', 'Challenged', 'Unchallenged', 'Mixed', 'Partially Verified', 'Not assessed', 'Reproduction in progress']:
        if category in sankey_detailed_mapping:
            # Count total for categories with subcategories
            total = 0
            
            # For Unchallenged, we need to include the reproducibility project items in the initial flow
            skip_types = []
            if category == 'Verified':
                skip_types = ['Verified by reproducibility project']
            elif category == 'Challenged':
                skip_types = ['Challenged by reproducibility project']
            
            for subcategory_name, subcategory_types in sankey_detailed_mapping[category].items():
                if subcategory_name != 'Selected for manual reproduction':  # Updated label
                    subcategory_types_filtered = [t for t in subcategory_types if t not in skip_types]
                    if subcategory_types_filtered:
                        mask = df['assessment_type'].isin(subcategory_types_filtered)
                        total += df[mask]['assessment_type'].count()
            
            # Include reproducibility project items in the initial flow to Unchallenged
            if category == 'Unchallenged':
                total += repro_total  # Add repro items to the initial unchallenged count
            
            if total > 0:
                first_level_counts[category] = total
                nodes.append(category)
                # Add percentage to first level node labels
                percentage = (total / total_claims) * 100
                node_labels.append(f'{category} ({total}, {percentage:.1f}%)')
                source.append(0)
                target.append(len(nodes) - 1)
                value.append(total)
                node_colors.append(base_colors.get(category, '#95a5a6'))
        else:
            # Direct count for categories without subcategories
            if category == 'Partially Verified':
                mask = df['assessment_type'] == 'Partially verified'
            elif category == 'Mixed':
                mask = df['assessment_type'] == 'Mixed'
            elif category == 'Not assessed':
                mask = df['assessment_type'] == 'Not assessed'
            elif category == 'Reproduction in progress':
                mask = df['assessment_type'] == 'Reproduction in progress'
            else:
                continue
                
            count = df[mask]['assessment_type'].count()
            if count > 0:
                nodes.append(category)
                # Add percentage to first level node labels
                percentage = (count / total_claims) * 100
                node_labels.append(f'{category} ({count}, {percentage:.1f}%)')
                source.append(0)
                target.append(len(nodes) - 1)
                value.append(count)
                node_colors.append(base_colors.get(category, '#95a5a6'))
    
    # Second level: detailed categories
    for main_category, subcategories in sankey_detailed_mapping.items():
        main_idx = nodes.index(main_category) if main_category in nodes else None
        if main_idx is not None:
            base_color = base_colors.get(main_category, '#95a5a6')
            for subcategory_name, assessment_types in subcategories.items():
                # Skip reproducibility project items for Verified/Challenged
                # as they'll be routed through Unchallenged first
                if (main_category in ['Verified', 'Challenged'] and 
                    subcategory_name in ['Verified by reproducibility project', 'Challenged by reproducibility project']):
                    continue
                
                # Get count
                mask = df['assessment_type'].isin(assessment_types)
                count = df[mask]['assessment_type'].count()
                
                # For "Selected for manual reproduction", use the precalculated counts
                if subcategory_name == 'Selected for manual reproduction':
                    count = repro_total
                
                if count > 0:
                    nodes.append(subcategory_name)
                    # Use custom node label if available, otherwise use the original name with count
                    display_name = node_label_mappings.get(subcategory_name, subcategory_name)
                    node_labels.append(f'{display_name} ({count})')
                    source.append(main_idx)
                    target.append(len(nodes) - 1)
                    value.append(count)
                    
                    # Use blue for the reproduction subcategory
                    if False: #subcategory_name == 'Selected for manual reproduction':
                        node_colors.append(reproduction_color)
                        link_colors.append(hex_to_rgba(reproduction_color))
                    else:
                        # Use lighter version of the base color for subcategories
                        node_colors.append(adjust_color(base_color, 0.85))
                        link_colors.append(hex_to_rgba(adjust_color(base_color, 0.85)))
    
    # Create initial link colors for other nodes
    # We need to skip the links we've already colored (for the reproduction pathway)
    link_count = len(source) - len(link_colors)
    for i in range(link_count):
        s = source[i]
        t = target[i]
        target_color = node_colors[t]
        link_colors.insert(i, hex_to_rgba(target_color))
    
    # Special case: Add flow from "Selected for manual reproduction" to direct endpoints
    if 'Unchallenged' in nodes and 'Selected for manual reproduction' in nodes:
        unchallenged_idx = nodes.index('Unchallenged')
        tested_idx = nodes.index('Selected for manual reproduction')
        
        if verified_repro_count > 0:
            # Add a node for "Verified" as a direct endpoint
            nodes.append('Verified by reproducibility project')
            display_name = node_label_mappings.get('Verified by reproducibility project', 'Verified by reproducibility project')
            node_labels.append(f'{display_name} ({verified_repro_count})')
            node_colors.append(ASSESSMENT_COLORS['Verified'])  # Use verified color
            
            # Add link from tested to verified - use blue to green gradient
            source.append(tested_idx)
            target.append(len(nodes) - 1)
            value.append(verified_repro_count)
            # Create a blend from blue to green for this link
            link_colors.append(hex_to_rgba(ASSESSMENT_COLORS['Verified']))
        
        if challenged_repro_count > 0:
            # Add a node for "Challenged" as a direct endpoint
            nodes.append('Challenged by reproducibility project')
            display_name = node_label_mappings.get('Challenged by reproducibility project', 'Challenged by reproducibility project')
            node_labels.append(f'{display_name} ({challenged_repro_count})')
            node_colors.append(ASSESSMENT_COLORS['Challenged'])  # Use challenged color
            
            # Add link from tested to challenged - use blue to red gradient
            source.append(tested_idx)
            target.append(len(nodes) - 1)
            value.append(challenged_repro_count)
            # Create a blend from blue to red for this link
            link_colors.append(hex_to_rgba(ASSESSMENT_COLORS['Challenged']))
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 14,
            line = dict(color = "black", width = 0.5),
            label = node_labels,
            color = node_colors,
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            color = link_colors
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text="Claims Assessment Flow",
        title_font_size=20,
        font_size=18,
        height=800,
        width=1200,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


# New: Three-layer Sankey diagram for claim assessment flow
def create_sankey_diagram2(df):
    """
    Build a three‑layer Sankey diagram that:
      • starts from “All Major Claims”,
      • shows detailed assessment categories in the first layer, and
      • collapses them into the final outcomes Verified / Challenged / Unchallenged.

    Layer 0 (root)
    └── “All Major Claims”

    Layer 1 (detailed leaves)
        – Verified in literature                (green)
        – Verified by same authors              (green)
        – Unchallenged, logically inconsistent  (grey)
        – Unchallenged, logically consistent    (grey)
        – Unchallenged                          (grey)
        – Selected for manual reproduction      (grey)  → splits later
        – Challenged in literature              (red)
        – Challenged by same authors            (red)
        – Partially Verified                    (yellow)
        – Mixed                                 (orange)

    Layer 2 (collapsed outcomes)
        – Verified     (green)
        – Challenged   (red)
        – Unchallenged (grey)

    The “Selected for manual reproduction” node splits into
    “Verified” and “Challenged” according to the experimental
    outcomes of the reproducibility project.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a column ``assessment_type`` with the
        detailed labels used in the ReproSci database.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    # ---------- helpers -------------------------------------------------
    def _cnt(labels):
        "Count rows whose assessment_type is in *labels*."
        return int(df["assessment_type"].isin(labels).sum())

    # ---------- raw counts ----------------------------------------------
    verified_lit              = _cnt(["Verified"])
    verified_same             = _cnt(["Verified by same authors"])
    unchall_consistent        = _cnt(["Unchallenged, logically consistent"])
    unchall_inconsistent      = _cnt(["Unchallenged, logically inconsistent"])
    unchall_general           = _cnt(["Unchallenged"])
    verified_repro            = _cnt(["Verified by reproducibility project"])
    challenged_repro          = _cnt(["Challenged by reproducibility project"])
    challenged_lit            = _cnt(["Challenged"])
    challenged_same           = _cnt(["Challenged by same authors"])
    partially_verified        = _cnt(["Partially verified"])
    mixed_cnt                 = _cnt(["Mixed"])

    selected_total            = verified_repro + challenged_repro
    total_claims = (
        verified_lit + verified_same
        + unchall_consistent + unchall_inconsistent + unchall_general
        + selected_total
        + challenged_lit + challenged_same
        + partially_verified + mixed_cnt
    )

    # ---------- node bookkeeping ----------------------------------------
    nodes, node_labels, node_colors = [], [], []
    source, target, value = [], [], []

    # root ---------------------------------------------------------------
    nodes.append("All Major Claims")
    node_labels.append(f"All Major Claims ({total_claims})")
    node_colors.append("#2c3e50")        # dark slate
    root_idx = 0

    # first‑layer specification -----------------------------------------
    first_layer_spec = [
        ("Verified in literature",             verified_lit,       ASSESSMENT_COLORS["Verified"]),
        ("Verified by same authors",           verified_same,      ASSESSMENT_COLORS["Verified"]),
        ("Unchallenged, logically inconsistent", unchall_inconsistent, ASSESSMENT_COLORS["Unchallenged"]),
        ("Unchallenged, logically consistent",   unchall_consistent,   ASSESSMENT_COLORS["Unchallenged"]),
        ("Unchallenged",                          unchall_general,     ASSESSMENT_COLORS["Unchallenged"]),
        ("Unchallenged selected for manual reproduction", selected_total,      adjust_color(ASSESSMENT_COLORS["Unchallenged"], 0.85)),
        ("Challenged in literature",             challenged_lit,      ASSESSMENT_COLORS["Challenged"]),
        ("Challenged by same authors",           challenged_same,     ASSESSMENT_COLORS["Challenged"]),
        ("Partially Verified",                  partially_verified,  ASSESSMENT_COLORS["Partially Verified"]),
        ("Mixed",                               mixed_cnt,          ASSESSMENT_COLORS["Mixed"]),
    ]

    first_layer_idx = {}
    for name, cnt, col in first_layer_spec:
        if cnt == 0:
            continue
        idx = len(nodes)
        first_layer_idx[name] = idx
        nodes.append(name)
        node_labels.append(f"{name} ({cnt})")
        node_colors.append(col)
        source.append(root_idx)
        target.append(idx)
        value.append(cnt)

    # outcome layer ------------------------------------------------------
    # totals for each final outcome
    verified_total     = verified_lit + verified_same + verified_repro
    challenged_total   = challenged_lit + challenged_same + challenged_repro
    unchallenged_total = unchall_inconsistent + unchall_consistent + unchall_general

    outcome_spec = [
        ("Verified",     verified_total,     ASSESSMENT_COLORS["Verified"]),
        ("Challenged",   challenged_total,   ASSESSMENT_COLORS["Challenged"]),
        ("Unchallenged", unchallenged_total, ASSESSMENT_COLORS["Unchallenged"]),
    ]
    outcome_idx = {}
    for name, total_cnt, col in outcome_spec:
        idx = len(nodes)
        outcome_idx[name] = idx
        nodes.append(name)
        node_labels.append(f"{name} ({total_cnt})")
        node_colors.append(col)

    # helper to add links ------------------------------------------------
    def _link(src_name, dst_name, cnt):
        if cnt == 0:
            return
        source.append(first_layer_idx[src_name])
        target.append(outcome_idx[dst_name])
        value.append(cnt)

    # verified flows (literature + same authors)
    _link("Verified in literature", "Verified", verified_lit)
    _link("Verified by same authors", "Verified", verified_same)

    # challenged flows (literature + same authors)
    _link("Challenged in literature", "Challenged", challenged_lit)
    _link("Challenged by same authors", "Challenged", challenged_same)

    # ── reproducibility‑project branch ────────────────────────────────
    selected_idx = first_layer_idx.get("Unchallenged selected for manual reproduction")

    if selected_idx is not None:
        # ▸ Verified by repro project
        if verified_repro > 0:
            ver_r_idx = len(nodes)
            nodes.append("Verified by ReproSci project")
            node_labels.append(f"Verified by ReproSci project ({verified_repro})")
            node_colors.append(ASSESSMENT_COLORS["Verified"])

            # selected → verified‑repro
            source.append(selected_idx)
            target.append(ver_r_idx)
            value.append(verified_repro)

            # verified‑repro → outcome Verified
            source.append(ver_r_idx)
            target.append(outcome_idx["Verified"])
            value.append(verified_repro)

        # ▸ Challenged by repro project
        if challenged_repro > 0:
            chal_r_idx = len(nodes)
            nodes.append("Challenged by ReproSci project")
            node_labels.append(f"Challenged by ReproSci project ({challenged_repro})")
            node_colors.append(ASSESSMENT_COLORS["Challenged"])

            # selected → challenged‑repro
            source.append(selected_idx)
            target.append(chal_r_idx)
            value.append(challenged_repro)

            # challenged‑repro → outcome Challenged
            source.append(chal_r_idx)
            target.append(outcome_idx["Challenged"])
            value.append(challenged_repro)

    # unchallenged flows
    _link("Unchallenged, logically inconsistent", "Unchallenged", unchall_inconsistent)
    _link("Unchallenged, logically consistent",   "Unchallenged", unchall_consistent)
    _link("Unchallenged",                         "Unchallenged", unchall_general)

    # colour each link with its destination node colour
    link_colors = [hex_to_rgba(node_colors[t]) for t in target]

    # build figure -------------------------------------------------------
    fig = go.Figure(go.Sankey(
        node = dict(
            pad = 40,
            thickness = 22,
            label = node_labels,
            color = node_colors,
            line  = dict(color="black", width=0.4),
        ),
        link = dict(
            source = source,
            target = target,
            value  = value,
            color  = link_colors,
        )
    ))

    fig.update_layout(
        title_text="",
        font_size=22,
        height=800,
        width=2000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    return fig




def create_horizontal_bar_chart(
    var_grouped,
    title,
    labels_map=None,
    show_p_value=True,
    other_n={},
    ax=None,
    orientation="horizontal",
    pct_axis_label="% of Claims",
    group_axis_label=None
):
    """
    Stacked bar chart of assessment categories, horizontal (default) or vertical.

    Parameters
    ----------
    var_grouped : DataFrame
        Index provides the groups to plot; columns like ``Challenged_prop`` etc.
    title : str
        Figure title.
    labels_map : dict or None
        Mapping from index values → pretty labels for the group axis.
    show_p_value : bool
        Placeholder for future statistical annotation (not yet implemented).
    other_n : dict
        {key: column_name} pairs whose sample sizes are appended to group labels.
    ax : matplotlib.axes.Axes or None
        Draw on this axes if provided; otherwise create a new figure.
    orientation : {"horizontal", "vertical"}
        Plot bars horizontally (previous behaviour) or vertically.
    pct_axis_label : str
        Label for the percentage axis.
    group_axis_label : str or None
        Label for the categorical axis (y-axis in horizontal, x-axis in vertical).

    Returns
    -------
    fig, ax : Matplotlib figure and axes.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure


    # Safe row ordering
# ------------------------------------------------------------------
    # If labels_map is provided, re-index rows to follow the key order.
    if labels_map is not None:
        # Handle boolean indices properly by creating a new DataFrame with explicit ordering
        available_keys = list(var_grouped.index)
        ordered_index = []
        for k in labels_map.keys():
            # Find matching keys using explicit equality for boolean values
            for idx in available_keys:
                if k == idx:  # This works correctly for boolean values
                    ordered_index.append(idx)
                    break
        if ordered_index:  # Only reorder if we found matching keys
            # Use reindex instead of loc to avoid boolean indexing issues
            var_grouped = var_grouped.reindex(ordered_index)

    # After any explicit re‑ordering, build the data matrix so that
    # numeric rows correspond one‑to‑one with var_grouped.index
    categories = ASSESSMENT_ORDER
    data = [var_grouped[f"{cat}_prop"].values for cat in categories]

    # ──────────────────────────────────────────────────────────────────
    if orientation == "horizontal":
        y = np.arange(len(var_grouped.index))
        left = np.zeros(len(var_grouped.index))
        bars = []

        for i, category in enumerate(categories):
            color = to_rgba(ASSESSMENT_COLORS[category], alpha=0.9)
            bar = ax.barh(
                y,
                data[i],
                left=left,
                label=category,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            left += data[i]
            bars.append(bar)

        # Data labels
        for bar_group in bars:
            for rect in bar_group:
                width = rect.get_width()
                if width > 0.02:
                    x_pos = rect.get_x() + width / 2
                    ax.text(
                        x_pos,
                        rect.get_y() + rect.get_height() / 2.0,
                        f"{width:.0%}" if width >= 0.1 else f"{width:.1%}",
                        ha="center",
                        va="center",
                        fontsize=SMALL_SIZE,
                        fontweight="bold",
                        color="white",
                    ).set_path_effects([withStroke(linewidth=3, foreground="black", alpha=0.7)])

        # Sample‑size text
        for i, lab in enumerate(var_grouped.index):
            sample_txt = f"$n_c$={var_grouped.loc[lab,'Major claims']}"
            ax.text(
                1.02,
                i,
                sample_txt,
                ha="left",
                va="center",
                fontsize=MEDIUM_SIZE, fontweight='normal',
                transform=ax.get_yaxis_transform(),
                usetex=True,
            )

        ax.set_yticks(y)

    # ──────────────────────────────────────────────────────────────────
    else:  # vertical
        x = np.arange(len(var_grouped.index))
        bottom = np.zeros(len(var_grouped.index))
        bars = []

        for i, category in enumerate(categories):
            color = to_rgba(ASSESSMENT_COLORS[category], alpha=0.9)
            bar = ax.bar(
                x,
                data[i],
                bottom=bottom,
                label=category,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += data[i]
            bars.append(bar)

        # Data labels
        for bar_group in bars:
            for rect in bar_group:
                height = rect.get_height()
                if height > 0.02:
                    y_pos = rect.get_y() + height / 2
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        y_pos,
                        f"{height:.0%}" if height >= 0.1 else f"{height:.1%}",
                        ha="center",
                        va="center",
                        fontsize=SMALL_SIZE,
                        fontweight="bold",
                        color="white",
                    ).set_path_effects([withStroke(linewidth=3, foreground="black", alpha=0.7)])

        # Sample‑size text above bars
        for i, lab in enumerate(var_grouped.index):
            sample_txt = f"$n_c$={var_grouped.loc[lab,'Major claims']}"
            ax.text(
                i,
                1.02,
                sample_txt,
                ha="center",
                va="bottom",
                fontsize=MEDIUM_SIZE, 
                fontweight='normal',
                transform=ax.get_xaxis_transform(),
                usetex=True,
            )

        ax.set_xticks(x)

    # ──────────────────────────────────────────────────────────────────
    # Build group labels (shared code)
    new_labels = []
    for lab in var_grouped.index:
        base_label = labels_map.get(lab, str(lab)) if labels_map else str(lab)
        for key, value in other_n.items():
            separator = "\n" if orientation == "horizontal" else "\n"
            #base_label += f"{separator}({var_grouped.loc[lab, value]} {key.lower()})"
            base_label += f"{separator}({var_grouped.loc[lab, value]})"
        new_labels.append(base_label)

    if orientation == "horizontal":
        ax.set_yticklabels(new_labels, fontweight="normal")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(pct_axis_label, fontweight="normal")
        if group_axis_label is not None:
            ax.set_ylabel(group_axis_label, fontweight="normal")
    else:
        ax.set_xticklabels(new_labels, fontweight="normal", rotation=0, ha="center")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(pct_axis_label, fontweight="normal")

    # Common styling
    ax.set_title(title, fontweight="bold", pad=20)

    if orientation == "horizontal":
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    else:  # vertical -> percentages are on y‑axis
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend
    legend = ax.legend(
        title="Assessment Category",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15) if orientation == "horizontal" else (1.02, 1),
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
        ncol=len(categories) if orientation == "horizontal" else 1,
    )
    legend.get_title().set_fontweight("bold")

    # Add x-axis label for vertical orientation
    if orientation == "vertical" and group_axis_label is not None:
        ax.set_xlabel(group_axis_label, fontweight="normal")

    plt.tight_layout()
    return fig, ax


def create_unified_legend(fig, axes_list, bbox_to_anchor=(0.5, 0.95), ncol=2, title="Assessment Category"):
    """
    Create a unified legend for composite figures with full control over placement.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to add the legend to
    axes_list : list
        List of axes to remove individual legends from
    bbox_to_anchor : tuple
        Position for the legend (x, y) in figure coordinates
    ncol : int
        Number of columns in the legend
    title : str or None
        Title for the legend (None for no title)
        
    Returns:
    --------
    legend : matplotlib.legend.Legend
        The created legend object
    """
    import matplotlib.patches as patches
    
    # Remove legends from all individual panels
    for ax in axes_list:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    
    # Create legend handles manually using assessment categories
    handles = []
    labels = ASSESSMENT_ORDER
    
    for category in labels:
        color = ASSESSMENT_COLORS[category]
        handle = patches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5)
        handles.append(handle)
    
    # Create and place the legend
    legend = fig.legend(
        handles,
        labels,
        title=title,
        loc="center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=True,
        fontsize=SMALL_SIZE,
        title_fontsize=MEDIUM_SIZE,
    )
    if title:  # Only set title properties if title is provided
        legend.get_title().set_fontweight('bold')
    
    return legend



# First author stuff


def plot_author_irreproducibility_focused(
    df,
    title="First Author Irreproducibility Distribution",
    fig_size=(10, 8),
    color_by='Unchallenged prop',
    cmap='viridis',
    most_challenged_on_right=True,
    name_col='Name',
    annotate_top_n=5,
    ax=None,
    show_stats_text=True,
):
    """
    Create a more focused Figure 4A showing the distribution of irreproducibility.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The prepared dataframe with author-level metrics
    title : str
        Plot title
    fig_size : tuple
        Figure size
    color_by : str
        Column name to use for point colors
    cmap : str or matplotlib colormap
        Colormap to use for points
    most_challenged_on_right : bool
        If True, place authors with highest challenged proportion on the right
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Create / reuse axes ------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Sort authors by challenged proportion
    sorting_col = 'Challenged prop'
    sorted_df = df.sort_values(sorting_col, ascending=not most_challenged_on_right).reset_index(drop=True)
    
    # Create rank column (1-indexed)
    sorted_df['Rank'] = np.arange(1, len(sorted_df) + 1)

        # Add size legend
    if "first_author_key" in df.columns:
        sizes = [6, 9, 12]
        scatter_size = 25
    else:
        sizes = [5, 10, 20, 50]
        scatter_size = 15
    
    # Add a line connecting the points
    ax.plot(
        sorted_df['Rank'],
        sorted_df['Challenged prop'],
        color='black',
        alpha=1,
        linestyle='-',
        linewidth=1
    )
    
    # Plot the distribution
    scatter = ax.scatter(
        sorted_df['Rank'],
        sorted_df['Challenged prop'],
        s=sorted_df['Major claims'] * scatter_size,  # Size by total claims
        c=sorted_df[color_by],  # Color by specified column
        cmap=cmap,
        alpha=0.9,
        edgecolors='white',
        linewidth=0
    )
    
    # Add color bar
    if most_challenged_on_right:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=-0.12)
    else:
        # Place colorbar on the left, inside the plot area
        cbar = plt.colorbar(
            scatter, ax=ax, shrink=0.6, pad=0.02, location='left',
            anchor=(3.5, 0.8), aspect=30, fraction=0.1
        )
    cbar.set_label(f'{color_by}', fontsize=MEDIUM_SIZE)
    
    for s in sizes:
        ax.scatter([], [], s=s*scatter_size, c='gray', alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=f'{s} Claims')
    
    # Label some notable authors (top 5 with highest proportion)
    top_authors = sorted_df.head(annotate_top_n)
    for _, row in top_authors.iterrows():
        ax.annotate(
            row[name_col],
            xy=(row['Rank'], row['Challenged prop']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    ## Add horizontal lines for reference
    #percentiles = [25, 50, 75]
    #for p in percentiles:
    #    val = np.percentile(df['Challenged prop'], p)
    #    ax.axhline(val, linestyle='--', color='gray', alpha=0.7,
    #            label=f'{p}th Percentile: {val:.1%}')
    
    # Set labels and title
    x_label = 'Author Rank'
    if most_challenged_on_right:
        x_label += ' (most to least challenged)'
    else:
        x_label += ' (least to most challenged)'
        
    ax.set_xlabel(x_label, fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Proportion of Challenged Claims', fontsize=MEDIUM_SIZE)
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add summary statistics as text
    stats_text = (
        f"Total Authors: {len(df)}\n"
        f"Mean: {df['Challenged prop'].mean():.1%}\n"
        f"Median: {df['Challenged prop'].median():.1%}\n"
        f"Authors with >20% Challenged: {(df['Challenged prop'] > 0.2).sum()}"
    )
    print(stats_text)
    if show_stats_text:
        ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=SMALL_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add legend
    ax.legend(
        title="Total Claims",
        loc='upper right',
        bbox_to_anchor=(0.65, 0.95),
        fontsize=MEDIUM_SIZE,
        title_fontsize=MEDIUM_SIZE,
        frameon=True,
    )
    
    # Set y-axis to start at 0
    #ax.set_ylim(0, None)
    #ax.set_xlim(1, None)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_challenged_histogram(
    df,
    prop_column='Challenged prop',
    title="Distribution of Challenged Claims Across First Authors",
    fig_size=(10, 6),
    color='#e74c3c',
    ax=None
):
    """
    Create a histogram with KDE showing the distribution of challenged claim proportions.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The prepared dataframe with author-level metrics
    prop_column : str
        Column name containing the proportion values to plot
    title : str
        Plot title
    fig_size : tuple
        Figure size
    color : str
        Color for the histogram bars
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Create histogram with KDE
    sns.histplot(
        data=df,
        x=prop_column,
        kde=True,
        bins=20,
        color=color,
        alpha=0.7,
        ax=ax
    )
    
    # Calculate summary statistics
    mean_val = df[prop_column].mean()
    median_val = df[prop_column].median()
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='black', linestyle='--', 
               label=f'Mean: {mean_val:.1%}')
    ax.axvline(median_val, color='black', linestyle='-', 
               label=f'Median: {median_val:.1%}')
    
    # Add stats information
    stats_text = (
        f"Total Authors: {len(df)}\n"
        f"Mean: {mean_val:.1%}\n"
        f"Median: {median_val:.1%}\n"
        f"Std Dev: {df[prop_column].std():.1%}\n"
        f"Authors with >20% Challenged: {(df[prop_column] > 0.2).sum()} ({(df[prop_column] > 0.2).sum()/len(df):.1%})"
    ) 
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
           fontsize=SMALL_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Format axes
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel('Proportion of Challenged Claims', fontweight='bold')
    ax.set_ylabel('Number of Authors', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig, ax

def plot_lorenz_curve(
    df,
    prop_column='Challenged prop',
    weight_column='Major claims',
    title="Distribution Inequality of Challenged Claims",
    fig_size=(10, 6),
    color='#e74c3c',
    ax=None,
    print_gini=True,
    print_top_txt=True
):
    """
    Create a Lorenz curve to visualize inequality in the distribution of challenged claims.
    Also calculates the Gini coefficient.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The prepared dataframe with author-level metrics
    prop_column : str
        Column name containing the proportion values to plot
    weight_column : str
        Column name containing weights (typically number of claims)
    title : str
        Plot title
    fig_size : tuple
        Figure size
    color : str
        Color for the curve
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Calculate contribution of each author to total challenged claims
    df = df.copy()
    df['challenged_count'] = df[prop_column] * df[weight_column]
    df['contribution'] = df['challenged_count'] / df['challenged_count'].sum()
    
    # Sort by contribution in DESCENDING order (highest contributors first)
    df = df.sort_values('contribution', ascending=False)
    
    # Calculate cumulative percentage of authors and challenged claims
    df['cum_authors'] = np.arange(1, len(df) + 1) / len(df)
    df['cum_challenged'] = df['contribution'].cumsum()
    
    # Now re-sort to get the proper Lorenz curve (lowest contributors first)
    df = df.sort_values('contribution')
    
    # Recalculate cumulative authors and challenged proportions for the curve
    df['lorenz_cum_authors'] = np.linspace(1/len(df), 1, len(df))
    df['lorenz_cum_challenged'] = df['contribution'].cumsum()
    
    # Plot Lorenz curve adding a 0
    #ax.plot(df['lorenz_cum_authors'], df['lorenz_cum_challenged'], color=color, linewidth=2.5, marker='o')
    ax.plot([0] + df['lorenz_cum_authors'].tolist(), [0] + df['lorenz_cum_challenged'].tolist(), 
        color=color, linewidth=2.5, marker='o')
    
    # Plot perfect equality line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect Equality')

    # Add shading between curve and equality line
    x_vals = [0] + df['lorenz_cum_authors'].tolist()
    y_vals = [0] + df['lorenz_cum_challenged'].tolist()
    ax.fill_between(x_vals, y_vals, x_vals, color="gray", alpha=0.1)
    
    # Calculate Gini coefficient
    # Area between perfect equality line and Lorenz curve / area under perfect equality line
    gini = 1 - 2 * np.trapz(df['lorenz_cum_challenged'], df['lorenz_cum_authors'])
    
    # Add Gini coefficient as text
    if print_gini:
        ax.text(0.05, 0.9, f"Gini Coefficient: {gini:.3f}", transform=ax.transAxes, ha='left',
            fontsize=MEDIUM_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add cumulative contribution lines
    n = len(df)
    sorted_df = df.sort_values('contribution', ascending=False)
    sorted_df['cum_contribution'] = sorted_df['contribution'].cumsum()

    top_10_pct = sorted_df['cum_contribution'].iloc[int(0.1*n)-1] if int(0.1*n) > 0 else 0
    top_20_pct = sorted_df['cum_contribution'].iloc[int(0.2*n)-1] if int(0.2*n) > 0 else 0
    top_50_pct = sorted_df['cum_contribution'].iloc[int(0.5*n)-1] if int(0.5*n) > 0 else 0
    
    # Add more detailed statistics
    stats_text = (
        f"Top 10% of authors account for {top_10_pct:.1%} of challenged claims\n"
        f"Top 20% of authors account for {top_20_pct:.1%} of challenged claims\n"
        f"Top 50% of authors account for {top_50_pct:.1%} of challenged claims"
    )

    for pct in [0.1, 0.2, 0.5]:
        idx = min(int((1-pct) * len(df)), len(df)-1)
        ax.scatter(df['lorenz_cum_authors'].iloc[idx], df['lorenz_cum_challenged'].iloc[idx], 
                color='black', zorder=3, s=40)
        ax.annotate(f"{pct:.0%}", 
                (df['lorenz_cum_authors'].iloc[idx], df['lorenz_cum_challenged'].iloc[idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=MEDIUM_SIZE)
    if print_top_txt:
        ax.text(0.05, 0.98, stats_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=SMALL_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    
    # Format axes
    ax.set_xlabel('Cumulative % of Authors', fontsize=MEDIUM_SIZE)#, fontweight='bold')
    ax.set_ylabel('Cumulative % of Challenged Claims', fontsize=MEDIUM_SIZE)#, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Format as percentages
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    return fig, ax





def create_publication_scatter(
    df, 
    x_var, 
    y_var, 
    group_by=None,
    min_articles=0,
    size_var=None, 
    log_scale=False,
    x_percent=False,
    y_percent=False,
    title=None,
    x_label=None,
    y_label=None,
    annotate_top_n=5,
    show_regression=True,
    fig_size=(12, 10),
    name_col='Name',
    ax=None
):
    """
    Create a publication-ready scatter plot for reproducibility analysis.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data to plot
    x_var : str
        Column name for x-axis variable
    y_var : str
        Column name for y-axis variable
    group_by : str, optional
        Column name to group and color by
    min_articles : int, default=1
        Minimum number of articles to include a point
    size_var : str, optional
        Column name for variable to determine point size
    log_scale : bool or str, default=False
        Use log scale for axes. Options: False, 'x', 'y', 'both'
    x_percent : bool, default=False
        Format x-axis as percentage
    y_percent : bool, default=False
        Format y-axis as percentage
    title : str, optional
        Plot title
    x_label : str, optional
        X-axis label (defaults to x_var if None)
    y_label : str, optional
        Y-axis label (defaults to y_var if None)
    annotate_top_n : int, default=5
        Number of points to label (by highest values)
    show_regression : bool, default=True
        Show regression line and statistics
    fig_size : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Filter data to include only rows with minimum number of articles
    if 'Articles' in df.columns:
        plot_df = df[df['Articles'] >= min_articles].copy()
    else:
        plot_df = df.copy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Prepare data
    x_data = plot_df[x_var]
    y_data = plot_df[y_var]
    
    # Determine point sizes
    if size_var is not None:
        sizes = plot_df[size_var]
        # Scale sizes for better visualization
        size_min, size_max = 50, 500
        if sizes.min() != sizes.max():  # Avoid division by zero
            scaled_sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min())) * (size_max - size_min) + size_min
        else:
            scaled_sizes = [size_min] * len(sizes)
    else:
        scaled_sizes = 100  # Default size
    
    # Determine colors based on grouping
    if group_by is not None:
        groups = plot_df[group_by].unique()
        cmap = plt.cm.tab10
        colors = {group: cmap(i % 10) for i, group in enumerate(groups)}
        
        # Plot each group separately for legend
        for group in groups:
            group_mask = plot_df[group_by] == group
            ax.scatter(
                x_data[group_mask], 
                y_data[group_mask],
                s=scaled_sizes if isinstance(scaled_sizes, int) else scaled_sizes[group_mask],
                c=[colors[group]],
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5,
                label=group
            )
    else:
        # Single group - use default color scheme
        ax.scatter(
            x_data, 
            y_data,
            s=scaled_sizes,
            c='#3498db',  # Default blue
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Set logarithmic scales if requested
    if log_scale == 'x' or log_scale == 'both':
        ax.set_xscale('log')
    if log_scale == 'y' or log_scale == 'both':
        ax.set_yscale('log')
    
    # Format axes as percentages if requested
    if x_percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if y_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add regression line if requested
    if show_regression:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        
        # Generate line points
        x_line = np.linspace(min(x_data), max(x_data), 100)
        y_line = intercept + slope * x_line
        
        # Plot regression line
        ax.plot(x_line, y_line, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add regression statistics text
        stats_text = f"$r^2$ = {r_value**2:.3f}\n"
        stats_text += f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
        stats_text += "\n" + f"y = {slope:.3f}x + {intercept:.3f}"
        
        # Position text in the upper left or upper right corner depending on slope
        if slope > 0:
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   va='top', ha='left', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
        else:
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   va='top', ha='right', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Annotate top points
    if annotate_top_n > 0:
        # Sort by y_var and get top N authors
        top_authors = plot_df.sort_values(by=y_var, ascending=False).head(annotate_top_n)
        
        for _, row in top_authors.iterrows():
            ax.annotate(
                row[name_col], 
                xy=(row[x_var], row[y_var]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    # Set labels and title
    ax.set_xlabel(x_label if x_label else x_var, fontweight='bold')
    ax.set_ylabel(y_label if y_label else y_var, fontweight='bold')
    ax.set_title(title if title else f"{y_var} vs {x_var}", fontweight='bold', pad=20)
    
    # Add legend if grouped
    if group_by is not None:
        legend = ax.legend(
            title=group_by,
            loc='best',
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray'
        )
        legend.get_title().set_fontweight('bold')
    
    # Add size legend if using size variable
    if size_var is not None and not isinstance(scaled_sizes, int):
        # Create a separate axis for size legend
        ax_legend = fig.add_axes([0.15, 0.55, 0.1, 0.2])
        ax_legend.axis('off')
        
        # Get values for legend
        size_values = [plot_df[size_var].min(), plot_df[size_var].max()/2, plot_df[size_var].max()]
        scaled_legend_sizes = ((np.array(size_values) - plot_df[size_var].min()) / 
                              (plot_df[size_var].max() - plot_df[size_var].min())) * (size_max - size_min) + size_min
        
        # Create size legend
        for i, (size, scaled_size) in enumerate(zip(size_values, scaled_legend_sizes)):
            ax_legend.scatter([], [], s=scaled_size, c='#3498db', alpha=0.7, 
                             edgecolors='white', linewidth=0.5, 
                             label=f"{int(size)}")
        
        ax_legend.legend(
            title=size_var,
            loc='center',
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray',
            labelspacing=2
        )
    
    # Customize grid
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax

def create_challenged_vs_unchallenged_scatter(
    plot_df,
    annotate_top_n=10,
    title="Challenged vs. Unchallenged Claims by Author",
    fig_size=(10, 8),
    size_mult=40,
    name_col='Name',
    ax=None
):
    """
    Create a scatter plot showing proportion of challenged vs. unchallenged claims by author.
    
    Parameters:
    -----------
    plot_df : pandas DataFrame
        The dataframe with author-level data
    annotate_top_n : int, default=10
        Number of authors to annotate
    title : str, default="Challenged vs. Unchallenged Claims by Author"
        Plot title
    fig_size : tuple, default=(10, 8)
        Figure size
    size_mult : int, default=40
        Multiplier for point sizes
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    print("Warning: this function plot the proportion of unchallenged in claims that are not challenged" \
        " and not the proportion of unchallenged claims in major claims. ")
    plot_df["Unchallenged_corrected"] = plot_df['Unchallenged'] / (plot_df['Major claims'] - plot_df['Challenged'])
    # Ensure no division by zero
    plot_df['Unchallenged_corrected'] = plot_df['Unchallenged_corrected'].replace([np.inf, -np.inf, np.nan], 0)

    # Scatter plot for challenged vs unchallenged
    scatter = ax.scatter(
        plot_df['Unchallenged_corrected'],
        plot_df['Challenged prop'],
        s=plot_df['Articles']*size_mult,  # Size by number of articles
        c=plot_df['Verified']/plot_df['Major claims'],  # Color by verification rate
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Annotate top authors
    top_authors = plot_df.sort_values(by='Challenged prop', ascending=False).head(annotate_top_n)
    
    for _, row in top_authors.iterrows():
        ax.annotate(
            row[name_col], 
            xy=(row['Unchallenged prop'], row['Challenged prop']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df['Unchallenged_corrected'], plot_df['Challenged prop']
    )
    
    # Plot regression line
    x_line = np.linspace(0, plot_df['Unchallenged_corrected'].max()*1.1, 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2)

    
    # Add regression statistics
    stats_text = f"$r^2$ = {r_value**2:.3f}\n"
    stats_text += f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            va='top', ha='left', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Customize plot
    ax.set_xlabel('Proportion of Unchallenged Claims in Claims not "Challenged"', fontweight='bold')
    ax.set_ylabel('Proportion of Challenged Claims', fontweight='bold')
    #ax.set_title(title, fontweight='bold')
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0)
    cbar.set_label(f'Proportion of Verified Claims', fontweight='bold')
    
    # Add size legend
    handles, labels = [], []
    if "first_author_key" in plot_df.columns:
        sizes = [1, 2, 3, 5]
    else:
        sizes = [3, 6, 12, 30]
    for size in sizes:
        handles.append(plt.scatter([], [], s=size*size_mult, color='gray', alpha=0.7))
        labels.append(f"{size}")
    legend = ax.legend(
        handles, labels,
        title="Number of Articles",
        loc='upper right',
        #frameon=True,
        framealpha=0.9,
        edgecolor='lightgray',
        handletextpad=2,
        labelspacing=1
    )
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    return fig, ax


def create_challenged_vs_articles_scatter(
    plot_df,
    annotate_top_n=10,
    title="Proportion of Challenged Claims vs. Number of Articles",
    fig_size=(10, 8),
    size_mult=40,
    name_col=None,
    ax=None
):
    """
    Create a scatter plot showing proportion of challenged claims vs. number of articles.
    
    Parameters:
    -----------
    plot_df : pandas DataFrame
        The dataframe with author-level data
    annotate_top_n : int, default=10
        Number of authors to annotate
    title : str, default="Proportion of Challenged Claims vs. Number of Articles"
        Plot title
    fig_size : tuple, default=(10, 8)
        Figure size
    size_mult : int, default=40
        Multiplier for point sizes
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Scatter plot
    scatter = ax.scatter(
        plot_df['Articles'], 
        plot_df['Challenged prop'],
        s=plot_df['Major claims']*size_mult,  # Size by number of claims
        c=plot_df['Verified']/plot_df['Major claims']*100,  # Color by verification rate
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5,
    )
    
    # Annotate top authors by percentage challenged
    top_pct_authors = plot_df.sort_values(by='Challenged prop', ascending=False).head(annotate_top_n)
    
    if name_col is not None:
        for _, row in top_pct_authors.iterrows():
            ax.annotate(
                row[name_col], 
                xy=(row['Articles'], row['Challenged prop']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df['Articles'], plot_df['Challenged prop']
    )
    
    # Plot regression line
    x_line = np.linspace(0, plot_df['Articles'].max()*1.1, 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color='#e74c3c', linestyle='-', alpha=1, linewidth=3)
    
    # Add regression statistics
    stats_text = f"$r^2$ = {r_value**2:.3f}\n"
    stats_text += f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            va='top', ha='left', fontsize=MEDIUM_SIZE, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Customize plot
    ax.set_xlabel('Number of Articles', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('% Challenged Claims', fontsize=MEDIUM_SIZE)
    #ax.set_title(title, fontweight='bold')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ## Add colorbar
    #cbar = plt.colorbar(
    #    plt.cm.ScalarMappable(
    #        norm=plt.Normalize(0, 1),
    #        cmap='viridis'
    #    ), 
    #    ax=ax
    #)
    #cbar.set_label('Proportion of Verified Claims', fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0)
    cbar.set_label(f'% Verified', fontsize=MEDIUM_SIZE)
    
    # Add size legend
    handles, labels = [], []
    if "first_author_key" in plot_df.columns:
        sizes = [2, 4, 6, 10]
    else:
        sizes = [6, 12, 20, 30, 60]
    for size in sizes:
        handles.append(plt.scatter([], [], s=size*size_mult, color='gray', alpha=0.7))
        labels.append(f"{size}")
    legend = ax.legend(
        handles, labels,
        title="Number of Claims",
        loc='upper right',
        #frameon=True,
        framealpha=0.9,
        edgecolor='lightgray',
        handletextpad=2,
        labelspacing=1,
        fontsize=MEDIUM_SIZE,
        frameon=True,
    )
    legend.get_title().set_fontsize(MEDIUM_SIZE)
    
    plt.tight_layout()
    
    return fig, ax