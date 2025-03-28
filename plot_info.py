import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np

# Set style parameters
plt.style.use('seaborn-v0_8-white')
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

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

## Verified                                 559
## Verified by reproducibility project        7
## Verified by same authors                  44
## Partially verified                        75
## Unchallenged, logically consistent       111
## Unchallenged                             107
## Unchallenged, logically inconsistent      22
## Mixed                                     12
## Challenged by reproducibility project     38
## Challenged                                26 


## Challenged by same authors                 5
    
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
    if impact_factor >= 30:
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
    print(len(filtered_df), len(df_copy))
    if len(filtered_df) != len(df_copy):
        print("🛑 missing rows:")
        print(df_copy[~(df_copy['group_by'].notna() & df_copy['assessment_group'].notna())][["assessment_group", "group_by"]])
    
    # Add standard category column for aggregation (needed for both expanded and non-expanded modes)
    filtered_df.loc[:, 'standard_category'] = filtered_df['assessment_group'].map(lambda x: category_mapping.get(x, None))
    
    # Create pivot tables for both detailed and standard categories
    detailed_pivot = pd.pivot_table(
        filtered_df,
        values='num',
        index='group_by',
        columns='assessment_group',
        aggfunc='count',
        fill_value=0
    )
    
    standard_pivot = pd.pivot_table(
        filtered_df,
        values='num',
        index='group_by',
        columns='standard_category',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder indices
    detailed_pivot = detailed_pivot.reindex(index=group_order)
    standard_pivot = standard_pivot.reindex(index=group_order)
    
    # Calculate the total counts for each group
    # Important: Use the standard categories pivot to ensure we don't double count
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
    category_positions = {
        cat: {
            'bottom': np.zeros(len(detailed_plot_data)),
            'height': np.zeros(len(detailed_plot_data))
        } for cat in set(category_mapping.values())
    }
    
    # Store handles for legend
    handles = []
    labels = []
    
    # Plot each assessment category in reverse order (for proper stacking)
    for cat in reversed(assessment_order):
        if cat in detailed_plot_data.columns:
            # Get standard category for this detailed category
            std_cat = category_mapping.get(cat)
            
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
    for std_cat in set(category_mapping.values()):
        if std_cat in standard_plot_data.columns:
            for i, group in enumerate(standard_plot_data.index):
                # Only add label if there's a non-zero value
                if standard_plot_data.loc[group, std_cat] > 0:
                    # Calculate position for label (center of the category's total height)
                    # Important: We need to recalculate the bottom and height for each category group
                    # This ensures labels are centered properly in each category section
                    
                    # For expanded categories, recalculate positions based on the detailed data
                    if use_expanded:
                        # Find which detailed categories belong to this standard category
                        detailed_cats = [cat for cat in assessment_order if category_mapping.get(cat) == std_cat]
                        
                        # Calculate the bottom position (min of all bottoms for this standard category)
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
                    count = int(standard_plot_data.loc[group, std_cat])
                    pct = standard_pct.loc[group, std_cat]
                    
                    if mode == 'absolute':
                        # For absolute mode, show only percentage
                        # Skip if percentage is too small
                        if pct <= 5:
                            continue
                        label_text = f'{pct:.1f}%'
                    else:
                        # For percentage mode, show only count
                        # Skip if count is too small
                        if count < 5:
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