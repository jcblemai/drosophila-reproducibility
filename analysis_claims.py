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

# %% [markdown]
# # Claim Analysis and Visualization

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patheffects import withStroke

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

# Define colors for assessment categories
ASSESSMENT_COLORS = {
    'Unchallenged': '#3498db',
    'Mixed': '#95a5a6',
    'Verified': '#2ecc71',
    'Partially Verified': '#f1c40f',
    'Challenged': '#e74c3c'
}

# Define the assessment category order for consistent plotting
ASSESSMENT_ORDER = ['Unchallenged', 'Mixed', 'Verified', 'Partially Verified', 'Challenged']

# Define detailed mapping for Sankey diagram
DETAILED_MAPPING = {
    'Verified': {
        'Verified by literature': ['Verified'],
        'Verified by reproducibility': ['Verified by reproducibility pr...'],
        'Verified by authors': ['Verified by same authors']
    },
    'Challenged': {
        'Challenged (general)': ['Challenged'],
        'Challenged by reproducibility': ['Challenged by reproducibility ...'],
        'Challenged by authors': ['Challenged by same authors']
    },
    'Unchallenged': {
        'Logically consistent': ['Unchallenged, logically consis...'],
        'Logically inconsistent': ['Unchallenged, logically incons...'],
        'General unchallenged': ['Unchallenged']
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

def group_assessment(assessment):
    """Group assessment types into broader categories."""
    if pd.isna(assessment) or assessment == 'Not assessed':
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
        return None

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


# %%
# Load data
df = pd.read_csv('preprocessed_data/claims_truncated_for_llm.csv')
df["assessment_type"].unique()

# %%
df[["assertion_type"]].value_counts()

# %%
df[["assessment_type"]].value_counts()

# %%
len(df)

# %%
major_claims_df = df[df['assertion_type'] == 'major_claim']
print(len(major_claims_df))
major_claims_df

# %%

# Apply categorizations
major_claims_df['journal_category'] = major_claims_df['impact_factor'].apply(categorize_journal)
major_claims_df['assessment_group'] = major_claims_df['assessment_type'].apply(group_assessment)

# %%
unique_pairs = major_claims_df[["journal_name", "impact_factor"]].drop_duplicates().sort_values("impact_factor", ascending=False)
for index, row in unique_pairs.iterrows():
    # Count occurrences of this journal in major_claims
    count = len(major_claims_df[major_claims_df["journal_name"] == row["journal_name"]])
    print(f"{row['impact_factor']:.1f}\t{row['journal_name']} ({count} claims)")

# %%
major_claims_df[["journal_category", "journal_name"]][major_claims_df["journal_category"] == "Trophy Journals"]["journal_name"].value_counts()


# %% [markdown]
# ## Constants and Helper Functions

# %%

# %% [markdown]
# ## Plot Functions

# %%
def create_stacked_bar_plot(df, mode='absolute', by_time=False):
    """
    Create a stacked bar plot of major claims.
    
    Parameters:
    - df: DataFrame with claims data
    - mode: 'absolute' for counts or 'percentage' for percentages
    - by_time: If True, group by time periods; if False, group by journal categories
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    
    # Apply categorizations
    df['assessment_group'] = df['assessment_type'].apply(group_assessment)
    
    if by_time:
        # Group by time periods
        df['group_by'] = df['year'].apply(bin_years)
        group_order = ['≤1991', '1992-1996', '1997-2001', '2002-2006', '2007-2011']
        x_label = 'Time Period'
    else:
        # Group by journal categories
        df['group_by'] = df['impact_factor'].apply(categorize_journal)
        group_order = ['Low Impact', 'High Impact', 'Trophy Journals']
        x_label = 'Journal Category'
    
    # Create pivot table
    pivot_data = pd.pivot_table(
        df[df['group_by'].notna() & df['assessment_group'].notna()],
        values='num',
        index='group_by',
        columns='assessment_group',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder index
    pivot_data = pivot_data.reindex(index=group_order)
    
    # Calculate percentages
    pivot_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose data based on mode
    plot_data = pivot_data if mode == 'absolute' else pivot_pct
    
    # Plot stacked bars
    bottom = np.zeros(len(plot_data))
    for col in ASSESSMENT_ORDER:
        if col in plot_data.columns:
            ax.bar(plot_data.index, plot_data[col], bottom=bottom, label=col, color=ASSESSMENT_COLORS[col])
            
            # Add labels
            for i in range(len(plot_data.index)):
                if plot_data[col][i] > 0:
                    # Format label based on mode
                    if mode == 'absolute':
                        label_text = f'{int(plot_data[col][i])} ({pivot_pct[col][i]:.1f}%)'
                    else:
                        # Only show labels for segments > 5% in percentage mode
                        if pivot_pct[col][i] <= 5:
                            continue
                        label_text = f'{pivot_pct[col][i]:.1f}%'
                    
                    text = ax.text(i, bottom[i] + plot_data[col][i]/2,
                                 label_text,
                                 ha='center', va='center',
                                 color='white', fontweight='bold')
                    text.set_path_effects([withStroke(linewidth=3, foreground='black')])
            
            bottom += plot_data[col]
    
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Customize grid
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
    
    ax.set_title(title, pad=20)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    
    # Rotate x-axis labels if needed
    if by_time:
        plt.xticks(rotation=45)
    
    # Add legend
    legend = ax.legend(title='Assessment Type',
                      bbox_to_anchor=(1.02, 0.5),
                      loc='center left',
                      fontsize=SMALL_SIZE,
                      title_fontsize=MEDIUM_SIZE)
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
    
    # Use global detailed mapping
    detailed_mapping = DETAILED_MAPPING
    
    # Count claims
    nodes = []
    node_labels = []
    source = []
    target = []
    value = []
    node_colors = []
    
    # Add root node
    total_claims = len(df)
    nodes.append('All Major Claims')
    node_labels.append(f'All Major Claims ({total_claims})')
    node_colors.append('#2c3e50')
    
    # First level: main categories
    first_level_counts = {}
    for category in ['Verified', 'Challenged', 'Unchallenged', 'Mixed', 'Partially Verified', 'Not assessed', 'Reproduction in progress']:
        if category in detailed_mapping:
            # Count total for categories with subcategories
            total = 0
            for subcategory_dict in detailed_mapping[category].values():
                mask = df['assessment_type'].isin(subcategory_dict)
                total += df[mask]['assessment_type'].count()
            
            if total > 0:
                first_level_counts[category] = total
                nodes.append(category)
                node_labels.append(f'{category} ({total})')
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
                node_labels.append(f'{category} ({count})')
                source.append(0)
                target.append(len(nodes) - 1)
                value.append(count)
                node_colors.append(base_colors.get(category, '#95a5a6'))
    
    # Second level: detailed categories
    for main_category, subcategories in detailed_mapping.items():
        main_idx = nodes.index(main_category) if main_category in nodes else None
        if main_idx is not None:
            base_color = base_colors.get(main_category, '#95a5a6')
            for subcategory_name, assessment_types in subcategories.items():
                mask = df['assessment_type'].isin(assessment_types)
                count = df[mask]['assessment_type'].count()
                
                if count > 0:
                    nodes.append(subcategory_name)
                    node_labels.append(f'{subcategory_name} ({count})')
                    source.append(main_idx)
                    target.append(len(nodes) - 1)
                    value.append(count)
                    # Use lighter version of the base color for subcategories
                    node_colors.append(adjust_color(base_color, 0.85))
    
    # Create link colors
    link_colors = []
    for s, t in zip(source, target):
        target_color = node_colors[t]
        link_colors.append(hex_to_rgba(target_color))
    
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

# %% [markdown]
# ## Analysis and Visualization

# %% [markdown]
# ### Journal Category Analysis

# %%
# Generate and save journal category plots
fig1, ax1 = create_stacked_bar_plot(major_claims_df, mode='absolute', by_time=False)
plt.savefig('figures/claims_journal_absolute.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/claims_journal_absolute.pdf', bbox_inches='tight')

fig2, ax2 = create_stacked_bar_plot(major_claims_df, mode='percentage', by_time=False)
plt.savefig('figures/claims_journal_percentage.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/claims_journal_percentage.pdf', bbox_inches='tight')

# %% [markdown]
# ### Time Period Analysis

# %%
# Generate and save time period plots
fig3, ax3 = create_stacked_bar_plot(major_claims_df, mode='absolute', by_time=True)
plt.savefig('figures/claims_time_absolute.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/claims_time_absolute.pdf', bbox_inches='tight')

fig4, ax4 = create_stacked_bar_plot(major_claims_df, mode='percentage', by_time=True)
plt.savefig('figures/claims_time_percentage.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/claims_time_percentage.pdf', bbox_inches='tight')

# %% [markdown]
# ### Trophy Journals Analysis

# %% [markdown]
# ## Sankey Diagram

# %%
# Create Sankey diagram
to_plot = major_claims_df[["assertion_type", "label", "assessment_type", "rank_assessment_type"]]
fig = create_sankey_diagram(to_plot)
fig.show()
fig.write_html('figures/claims_sankey.html')
