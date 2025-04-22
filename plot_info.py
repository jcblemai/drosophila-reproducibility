import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
from matplotlib.ticker import PercentFormatter, MaxNLocator
import seaborn as sns


import matplotlib.gridspec as gridspec

# Custom colors for categories

# Set style parameters
plt.style.use('seaborn-v0_8-white')
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

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

def create_stacked_bar_plot(df, mode='absolute', by_time=False, use_expanded=False):
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
        df_copy.loc[:, 'assessment_group'] = df_copy['assessment_type'].apply(group_detailed_assessment)
        assessment_order = ASSESSMENT_ORDER_EXPANDED
        assessment_colors = ASSESSMENT_COLORS_EXPANDED
        # Category mapping is globally defined for expanded mode
    else:
        df_copy.loc[:, 'assessment_group'] = df_copy['assessment_type'].apply(group_assessment)
        assessment_order = ASSESSMENT_ORDER
        assessment_colors = ASSESSMENT_COLORS
        category_mapping = {cat: cat for cat in ASSESSMENT_ORDER}  # Identity mapping
    
    if by_time:
        # Group by time periods
        df_copy.loc[:, 'group_by'] = df_copy['year'].apply(bin_years)
        group_order = ['≤1991', '1992-1996', '1997-2001', '2002-2006', '2007-2011']
        x_label = 'Time Period'
    else:
        # Group by journal categories
        df_copy.loc[:, 'group_by'] = df_copy['impact_factor'].apply(categorize_journal)
        group_order = ['Low Impact', 'High Impact', 'Trophy Journals']
        x_label = 'Journal Category'
    
    # Filter out rows with missing values
    filtered_df = df_copy[df_copy['group_by'].notna() & df_copy['assessment_group'].notna()].copy()
    print(f"Using {len(filtered_df)} of {len(df_copy)} rows")
    if len(filtered_df) != len(df_copy):
        print("🛑 missing rows:")
        print(df_copy[~(df_copy['group_by'].notna() & df_copy['assessment_group'].notna())][["assessment_group", "group_by"]])
    
    # Add standard category column for aggregation
    filtered_df.loc[:, 'standard_category'] = filtered_df['assessment_group'].map(lambda x: category_mapping.get(x, x))
    
    # Create pivot tables
    detailed_pivot = pd.pivot_table(
        filtered_df,
        values='article_id',
        index='group_by',
        columns='assessment_group',
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

    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
                color=assessment_colors[cat]
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
                            bottom_pos = category_positions[std_cat]['bottom'][i]
                            height = category_positions[std_cat]['height'][i]
                        
                        # Skip if height is zero (avoid division by zero)
                        if height <= 0:
                            continue
                        
                        # Calculate center position
                        center_pos = bottom_pos + (height / 2)
                        
                        # Get the percentage for this category
                        pct = standard_pct.loc[group, std_cat]
                        count = int(standard_pivot.loc[group, std_cat])
                        
                        # Skip small categories except for 'Challenged'
                        if pct <= 5 and std_cat != 'Challenged':
                            continue
                            
                        # Format as percentage for the percentage plot
                        label_text = f'{pct:.1f}%'
                        
                        # Add the text label
                        text = ax.text(
                            i, center_pos,
                            label_text,
                            ha='center', va='center',
                            color='white', fontweight='bold'
                        )
                        text.set_path_effects([withStroke(linewidth=3, foreground='black')])
    
        # Add total counts at the top of each column
        for i, group in enumerate(group_order):
            if group in row_totals.index:
                total = int(row_totals[group])
                # Position the text slightly above the top of the bar
                top_pos = bottom[i] + 1  # Add a small offset
                ax.text(
                    i, top_pos,
                    f'n={total}',
                    ha='center', va='bottom',
                    fontweight='bold',
                    fontsize=MEDIUM_SIZE
                )
    
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Set titles and labels
    if mode == 'absolute':
        y_label = 'Number of Claims'
        title_prefix = ''
    else:
        y_label = 'Percentage of Claims'
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
    
    # Ensure y-axis has enough room for the total count labels
    y_max = max(bottom) * 1.1  # Add 10% padding
    ax.set_ylim(0, y_max)
    
    # Rotate x-axis labels if needed
    if by_time:
        plt.xticks(rotation=45)
    
    # Reverse handles and labels to match the original order (not reversed)
    handles = handles[::-1]
    labels = labels[::-1]
    
    # Add legend with corrected order
    legend = ax.legend(
        handles=handles,
        labels=labels,
        title='Assessment Type',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        fontsize=SMALL_SIZE,
        title_fontsize=MEDIUM_SIZE
    )
    legend.get_frame().set_linewidth(0.0)
    
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
                    if subcategory_name == 'Selected for manual reproduction':
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
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = node_labels,
            color = node_colors
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
        font_size=14,
        height=800,
        width=1200,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig




def create_horizontal_bar_chart(var_grouped, title, labels, show_p_value=True, other_n={}):
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data for stacking
    categories = ['Verified', 'Partially Verified', 'Unchallenged', 'Mixed', 'Challenged']
    categories = ASSESSMENT_ORDER
    data = []
    for cat in categories:
        data.append(var_grouped[f'{cat}_prop'].values)
    
    # Create stacked bars horizontally
    y = np.arange(len(var_grouped.index))
    left = np.zeros(len(var_grouped.index))
    bars = []
    
    for i, category in enumerate(categories):
        color = to_rgba(ASSESSMENT_COLORS[category], alpha=0.9)
        bar = ax.barh(y, data[i], left=left, label=category, 
                    color=color, edgecolor='white', linewidth=0.5)
        left += data[i]
        bars.append(bar)
    
    # Add data labels to each segment
    for i, bar_group in enumerate(bars):
        for j, rect in enumerate(bar_group):
            width = rect.get_width()
            if width > 0.02:  # Only add label if segment is large enough
                # Position text in center of bar segment
                x_pos = rect.get_x() + width/2
                
                # Format percentage with proper precision
                text = ax.text(x_pos, rect.get_y() + rect.get_height()/2.,
                        f'{width:.0%}' if width >= 0.1 else f'{width:.1%}', 
                        ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white')
                
                # Add stroke effect for better readability
                text.set_path_effects([withStroke(linewidth=3, foreground='black', alpha=0.7)])
    
    # Add sample size as text to the right of each bar
    for i, lab in enumerate(var_grouped.index):
        str_to_plot = f"N_claims={var_grouped.loc[lab, 'Major claims']}"
        #for key, value in other_n.items():
        #    str_to_plot += f"\nn_{key}={var_grouped.loc[lab, value]}"
        ax.text(1.02, i, str_to_plot, 
                ha='left', va='center', fontsize=12, color='black',
                transform=ax.get_yaxis_transform())
    
    # Customize plot appearance
    ax.set_yticks(y)

    new_labels = []

    for i, lab in enumerate(var_grouped.index):
        new_labels.append(labels[i])
        for key, value in other_n.items():
            print(labels[i])
            print(var_grouped.loc[lab, value])
            print(i)
            new_labels[i] = labels[i] + f"\n(n={var_grouped.loc[lab, value]})"

    ax.set_yticklabels(new_labels, 
                fontweight='bold')
    ax.set_xlim(0, 1.05)  # Leave space for bar labels
    ax.set_xlabel('Proportion of Claims', fontweight='bold')
    ax.set_title(title, 
                 fontweight='bold', pad=20)
    
    # Improve x-axis formatting (as percentages)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    
    # Customize grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Enhance spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Add statistical annotation if requested
    if show_p_value:
        raise ValueError("This function is not yet implemented")
        t_stat, p_val = stats.ttest_ind(
            df[df['Historical lab'] == True]['reproducibility_score'].dropna(),
            df[df['Historical lab'] == False]['reproducibility_score'].dropna(),
            equal_var=False
        )
        
        # Format p-value with appropriate notation
        if p_val < 0.001:
            p_text = "p < 0.001***"
        elif p_val < 0.01:
            p_text = f"p = {p_val:.3f}**"
        elif p_val < 0.05:
            p_text = f"p = {p_val:.3f}*"
        else:
            p_text = f"p = {p_val:.3f} (ns)"
        
        # Add t-test result above the plot
        ax.text(0.5, 1.00, f"Reproducibility score comparison: t = {t_stat:.2f}, {p_text}", 
              ha='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Add legend with improved positioning and appearance
    legend = ax.legend(
        title="Assessment Category",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        frameon=True,
        framealpha=0.9,
        edgecolor='lightgray',
        ncol=len(categories)
    )
    legend.get_title().set_fontweight('bold')
    
    # Tight layout for better use of space
    plt.tight_layout()
    
    return fig, ax



# First author stuff


def plot_author_irreproducibility_focused(
    df,
    title="First Author Irreproducibility Distribution",
    fig_size=(10, 8),
    color_by='Unchallenged prop',
    cmap='viridis',
    most_challenged_on_right=True
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
    # Create a figure with one main plot
    fig, ax = plt.subplots(figsize=fig_size)
    
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
    
    
    # Plot the distribution
    scatter = ax.scatter(
        sorted_df['Rank'],
        sorted_df['Challenged prop'],
        s=sorted_df['Major claims'] * scatter_size,  # Size by total claims
        c=sorted_df[color_by],  # Color by specified column
        cmap=cmap,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add a line connecting the points
    ax.plot(
        sorted_df['Rank'],
        sorted_df['Challenged prop'],
        color='gray',
        alpha=0.5,
        linestyle='-',
        linewidth=1
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=-0.12)
    cbar.set_label(f'{color_by}', fontweight='bold')
    
    for s in sizes:
        ax.scatter([], [], s=s*scatter_size, c='gray', alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=f'{s} Claims')
    
    # Label some notable authors (top 5 with highest proportion)
    top_authors = sorted_df.head(5)
    for _, row in top_authors.iterrows():
        ax.annotate(
            row['Name'],
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
        x_label += ' (least to most challenged)'
    else:
        x_label += ' (most to least challenged)'
        
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel('Proportion of Challenged Claims', fontweight='bold')
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
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=SMALL_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add legend
    ax.legend(title="Total Claims", loc='upper right', bbox_to_anchor=(0.80, 0.65))
    
    # Set y-axis to start at 0
    ax.set_ylim(0, None)
    ax.set_xlim(1, None)
    
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
    color='#e74c3c'
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
    fig, ax = plt.subplots(figsize=fig_size)
    
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
    color='#e74c3c'
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
    fig, ax = plt.subplots(figsize=fig_size)
    
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
    ax.text(0.05, 0.75, f"Gini Coefficient: {gini:.3f}", transform=ax.transAxes, ha='left',
           fontsize=SMALL_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
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
                xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    ax.text(0.05, 0.98, stats_text, transform=ax.transAxes, ha='left', va='top',
           fontsize=SMALL_SIZE, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Format axes
    ax.set_xlabel('Cumulative Proportion of Authors', fontweight='bold')
    ax.set_ylabel('Cumulative Proportion of Challenged Claims', fontweight='bold')
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
    min_articles=1,
    size_var=None, 
    log_scale=False,
    x_percent=False,
    y_percent=False,
    title=None,
    x_label=None,
    y_label=None,
    annotate_top_n=5,
    show_regression=True,
    fig_size=(12, 10)
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
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
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
    if annotate_top_n > 0 and 'Name' in plot_df.columns:
        # Sort by y_var and get top N authors
        top_authors = plot_df.sort_values(by=y_var, ascending=False).head(annotate_top_n)
        
        for _, row in top_authors.iterrows():
            ax.annotate(
                row['Name'], 
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
        ax_legend = fig.add_axes([0.85, 0.15, 0.1, 0.2])
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
    size_mult=40
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
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Scatter plot for challenged vs unchallenged
    scatter = ax.scatter(
        plot_df['Unchallenged prop'], 
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
            row['Name'], 
            xy=(row['Unchallenged prop'], row['Challenged prop']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df['Unchallenged prop'], plot_df['Challenged prop']
    )
    
    # Plot regression line
    x_line = np.linspace(0, plot_df['Unchallenged prop'].max()*1.1, 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add regression statistics
    stats_text = f"$r^2$ = {r_value**2:.3f}\n"
    stats_text += f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            va='top', ha='left', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Customize plot
    ax.set_xlabel('Proportion of Unchallenged Claims', fontweight='bold')
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
    size_mult=40
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
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Scatter plot
    scatter = ax.scatter(
        plot_df['Articles'], 
        plot_df['Challenged prop'],
        s=plot_df['Major claims']*size_mult,  # Size by number of claims
        c=plot_df['Verified']/plot_df['Major claims'],  # Color by verification rate
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Annotate top authors by percentage challenged
    top_pct_authors = plot_df.sort_values(by='Challenged prop', ascending=False).head(annotate_top_n)
    
    for _, row in top_pct_authors.iterrows():
        ax.annotate(
            row['Name'], 
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
    ax.plot(x_line, y_line, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add regression statistics
    stats_text = f"$r^2$ = {r_value**2:.3f}\n"
    stats_text += f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            va='top', ha='left', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # Customize plot
    ax.set_xlabel('Number of Articles', fontweight='bold')
    ax.set_ylabel('Proportion of Challenged Claims', fontweight='bold')
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
    cbar.set_label(f'Proportion of Verified Claims', fontweight='bold')
    
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
        labelspacing=1
    )
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    return fig, ax