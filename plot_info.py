import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from scipy import stats
import matplotlib as mpl
from matplotlib.colors import to_rgba
import plotly.graph_objects as go


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
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

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
    if 'Verified' in str(assessment):
        return 'Verified'
    elif 'Challenged' in str(assessment):
        return 'Challenged'
    elif 'Mixed' in str(assessment):
        return 'Mixed'
    elif 'Partially verified' in str(assessment):
        return 'Partially Verified'
    elif 'Unchallenged' in str(assessment):
        return 'Unchallenged'
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
    
    # Add labels for standard categories
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
                    
                    # Format label based on mode
                    count = int(standard_pivot.loc[group, std_cat])  # Always get the actual count
                    pct = standard_pct.loc[group, std_cat]  # Always get the percentage
                    
                    if mode == 'absolute':
                        # For absolute mode, show percentage
                        if pct <= 10 and std_cat != 'Challenged':
                            continue
                        label_text = f'{pct:.1f}%'
                    else:
                        # For percentage mode, show count (n=) as requested
                        if count < 10 and std_cat != 'Challenged':
                            continue
                        label_text = f'n={count}'
                    
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
    
    ax.set_title(title, pad=20)
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


def create_horizontal_bar_chart(var_grouped, title, labels, show_p_value=True):
    
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
            if width > 0.04:  # Only add label if segment is large enough
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
        ax.text(1.02, i, f"n={var_grouped.loc[lab, 'Major claims']}", 
                ha='left', va='center', fontsize=12, color='black',
                transform=ax.get_yaxis_transform())
    
    # Customize plot appearance
    ax.set_yticks(y)
    ax.set_yticklabels(labels, 
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