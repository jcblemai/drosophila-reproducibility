# %% [markdown]
#  # Claim Analysis and Visualization

# %% [markdown]
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patheffects import withStroke
import plot_info
import wrangling

# %%
# Load data
df = pd.read_csv('preprocessed_data/claims_db_truncated_for_llm.csv')
df["assessment_type"].unique()

# %%
df[["assertion_type"]].value_counts()

# %%
major_claims_df = df[df['assertion_type'] == 'major_claim']
print(len(major_claims_df))
major_claims_df

# %%
major_claims_df[["year"]].value_counts()

# %%

# Display distribution of journal categories
print(f"Journal Category Distribution:")
print(major_claims_df['journal_category'].value_counts())
print("\nAssessment Group Distribution:")
print(major_claims_df['assessment_type_grouped'].value_counts())


# %%
unique_pairs = major_claims_df[["journal_name", "impact_factor"]].drop_duplicates().sort_values("impact_factor", ascending=False)
for index, row in unique_pairs.iterrows():
    # Count occurrences of this journal in major_claims
    count = len(major_claims_df[major_claims_df["journal_name"] == row["journal_name"]])
    print(f"{row['impact_factor']:.1f}\t{row['journal_name']} ({count} claims)")


# %%
major_claims_df

# %%
major_claims_df[["journal_category", "journal_name"]][major_claims_df["journal_category"] == "Trophy Journals"]["journal_name"].value_counts()

# %% [markdown]
#  ## Plot Functions

# %% [markdown]
#  ## Analysis and Visualization

# %% [markdown]
#  ### Journal Category Analysis

# %%
def create_journal_claims_table(df):
    """
    Create a table with journal name, impact factor, and counts for each claim type
    
    Parameters:
    - df: DataFrame containing the claims data
    
    Returns:
    - A DataFrame with journal information and claim counts
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Map detailed assessment types to standardized categories
    df_copy['standard_category'] = df_copy['assessment_type'].map(
        lambda x: plot_info.category_mapping.get(x, x)
    ) # TODO This mapping is not waht shoudl be used for the table
    
    # Group by journal name and impact factor
    grouped = df_copy.groupby(['journal_name', 'impact_factor'])['standard_category'].value_counts().unstack().fillna(0)
    
    # Convert counts to integers
    for col in grouped.columns:
        grouped[col] = grouped[col].astype(int)
    
    # Create a total count column
    grouped['Total'] = grouped.sum(axis=1)
    
    # Sort by impact factor (descending)
    grouped = grouped.sort_values(by='impact_factor', ascending=False)
    
    # Ensure all required columns exist
    required_columns = ['Challenged', 'Mixed', 'Partially Verified', 'Unchallenged', 'Verified', 'Total']
    for col in required_columns:
        if col not in grouped.columns:
            grouped[col] = 0
    
    # Reorder columns
    grouped = grouped[required_columns]
    
    # Reset index to make journal_name and impact_factor regular columns
    grouped = grouped.reset_index()
    
    return grouped

# Apply the function to generate the table
journal_claims_table = create_journal_claims_table(major_claims_df)

journal_claims_table.to_csv("figures/tableS1_claims_by_journal.csv", index=False)


# %%


# %%
# Generate and save journal category plots
fig1, ax1 = plot_info.create_stacked_bar_plot(major_claims_df, mode='absolute', by_time=False, use_expanded=False)
plt.savefig('figures/fig2A_claims_journal_absolute.png', dpi=300, bbox_inches='tight')

#plt.savefig('figures/fig2_claims_journal_absolute.pdf', bbox_inches='tight')

fig2, ax2 = plot_info.create_stacked_bar_plot(major_claims_df, mode='percentage', by_time=False, use_expanded=False)
plt.savefig('figures/fig2B_claims_journal_percentage.png', dpi=300, bbox_inches='tight')
#plt.savefig('figures/fig2_claims_journal_percentage.pdf', bbox_inches='tight')



# %% [markdown]
#  ### Time Period Analysis

# %%
# Generate and save time period plots
fig3, ax3 = plot_info.create_stacked_bar_plot(major_claims_df, mode='absolute', by_time=True)
plt.savefig('figures/fig3A_claims_time_absolute.png', dpi=300, bbox_inches='tight')
#plt.savefig('figures/claims_time_absolute.pdf', bbox_inches='tight')

fig4, ax4 = plot_info.create_stacked_bar_plot(major_claims_df, mode='percentage', by_time=True)
plt.savefig('figures/fig3B_claims_time_percentage.png', dpi=300, bbox_inches='tight')
#plt.savefig('figures/claims_time_percentage.pdf', bbox_inches='tight')


# %% [markdown]
#  ### Trophy Journals Analysis

# %% [markdown]
#  ## Figure 1 Sankey Diagra,

# %%


# %%
# Create Sankey diagram
to_plot = major_claims_df[["assertion_type", "label", "assessment_type", "rank_assessment_type"]]
fig = plot_info.create_sankey_diagram(to_plot)
fig.show()
fig.write_html('figures/fig1_claims_sankey.html')

# %%
fig = plot_info.create_sankey_diagram2(to_plot)
fig.show()
fig.write_html('figures/fig1_claims_sankey.html')

# %%
major_claims_df

# %%
assessment_columns = plot_info.assessment_columns

country_metrics = wrangling.create_author_metric(major_claims_df, variable='country')
country_metrics

# %%
to_plot = country_metrics

# %%


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl

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

# Set figure DPI for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Custom colors for categories
ASSESSMENT_COLORS = {
    'Challenged': '#e74c3c',       # Red
    'Mixed': '#f39c12',            # Orange
    'Unchallenged': '#95a5a6',     # Gray
    'Partially Verified': '#f1c40f', # Yellow
    'Verified': '#2ecc71'          # Green
}

def create_simple_horizontal_country_chart(
    df,
    country_col='country',
    var_to_plot='Challenged_prop',
    min_claims=0,
    sort_by_value=True,
    title=None,
    x_label=None,
    annotate_counts=True,
    fig_size=(12, 8)
):
    """
    Create a simple horizontal bar chart showing a variable by country.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with country-level data
    country_col : str, default='Country'
        Column name for country
    var_to_plot : str, default='Challenged_prop'
        Column to plot (e.g., 'Challenged_prop', 'Verified_prop')
    min_claims : int, default=5
        Minimum number of claims for a country to be included
    sort_by_value : bool, default=True
        If True, sort countries by the value being plotted
    title : str, optional
        Plot title
    x_label : str, optional
        X-axis label
    annotate_counts : bool, default=True
        Whether to add count annotations to bars
    fig_size : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Filter data
    filtered_df = df[df['Major claims'] >= min_claims].copy()
    
    # Determine if it's a percentage variable
    is_percent = '_prop' in var_to_plot
    
    # Aggregate data by country
    if is_percent:
        # For proportion variables, we need to calculate weighted average
        num_var = var_to_plot.replace('_prop', '')
        claims_by_country = filtered_df.groupby(country_col)['Major claims'].sum()
        counts_by_country = filtered_df.groupby(country_col)[num_var].sum()
        
        country_data = pd.DataFrame({
            'Major claims': claims_by_country,
            var_to_plot: counts_by_country / claims_by_country
        }).reset_index()
    else:
        # For count variables, we can just sum
        country_data = filtered_df.groupby(country_col).agg({
            var_to_plot: 'sum',
            'Major claims': 'sum'
        }).reset_index()
    
    # Sort data if requested
    if sort_by_value:
        country_data = country_data.sort_values(by=var_to_plot, ascending=False)
    else:
        country_data = country_data.sort_values(by=country_col)
    
    # Adjust figure height based on number of countries
    country_count = len(country_data)
    if country_count < 5:
        fig_size = (fig_size[0], max(4, country_count * 0.8))
    elif country_count > 12:
        fig_size = (fig_size[0], min(16, country_count * 0.6))
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get color based on variable
    if 'Challenged' in var_to_plot:
        color = ASSESSMENT_COLORS['Challenged']
    elif 'Verified' in var_to_plot:
        color = ASSESSMENT_COLORS['Verified']
    elif 'Unchallenged' in var_to_plot:
        color = ASSESSMENT_COLORS['Unchallenged']
    elif 'Mixed' in var_to_plot:
        color = ASSESSMENT_COLORS['Mixed']
    elif 'Partially Verified' in var_to_plot:
        color = ASSESSMENT_COLORS['Partially Verified']
    else:
        color = '#3498db'  # Default blue
    
    # Plot horizontal bars
    bars = ax.barh(
        country_data[country_col],
        country_data[var_to_plot],
        color=color,
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        
        # Format label based on whether it's a percentage
        if is_percent:
            label = f"{width:.1%}"
        else:
            label = f"{width:.0f}"
        
        ax.text(
            width/2,
            bar.get_y() + bar.get_height()/2,
            label,
            ha='center', va='center',
            color='white', fontweight='bold', fontsize=12
        )
    
    # Add count annotations if requested
    if annotate_counts:
        for i, (_, row) in enumerate(country_data.iterrows()):
            ax.text(
                0.01,
                i,
                f"n={int(row['Major claims'])}",
                va='center', ha='left',
                fontsize=10, color='black',
                transform=ax.get_yaxis_transform()
            )
    
    # Format x-axis as percentage if appropriate
    if is_percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add global mean line
    mean_val = country_data[var_to_plot].mean()
    if is_percent:
        mean_label = f"Mean: {mean_val:.1%}"
    else:
        mean_label = f"Mean: {mean_val:.1f}"
    
    ax.axvline(
        mean_val,
        color='black',
        linestyle='--',
        alpha=0.7,
        linewidth=1.5,
        label=mean_label
    )
    
    # Set labels and title
    if x_label:
        ax.set_xlabel(x_label, fontweight='bold')
    else:
        # Generate a reasonable x-label from the variable name
        if is_percent:
            x_label = f"Proportion of {var_to_plot.replace('_prop', '')} Claims"
        else:
            x_label = f"Number of {var_to_plot} Claims"
        ax.set_xlabel(x_label, fontweight='bold')
    
    ax.set_ylabel('Country', fontweight='bold')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    else:
        if is_percent:
            title = f"Proportion of {var_to_plot.replace('_prop', '')} Claims by Country"
        else:
            title = f"{var_to_plot} Claims by Country"
        ax.set_title(title, fontweight='bold', pad=20)
    
    # Customize grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='lightgray')
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax


# Example usage for simple chart:
fig, ax = create_simple_horizontal_country_chart(
    df=to_plot,
    var_to_plot='Challenged_prop',
    min_claims=5,
    title='Proportion of Challenged Claims by Country',
    x_label='Proportion of Challenged Claims'
)

# plt.savefig('country_comparison.png', dpi=300, bbox_inches='tight')

# %%
# Import necessary modules
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import matplotlib.pyplot as plt

# Define colors for categories
ASSESSMENT_COLORS = plot_info.ASSESSMENT_COLORS

# Define the category order for consistent stacking
ASSESSMENT_ORDER = plot_info.ASSESSMENT_ORDER


def create_two_panel_country_chart(
    df,
    country_col='country',
    min_claims=0,
    sort_by='Challenged_prop',
    title="Reproducibility Analysis by Country",
    fig_size=(15, 10)
):
    """
    Create a two-panel chart with stacked proportions and a separate bar chart
    for number of claims by country.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with country-level data
    country_col : str, default='country'
        Column name for country
    min_claims : int, default=5
        Minimum number of claims for a country to be included
    sort_by : str, default='Challenged_prop'
        How to sort countries
    title : str, default="Reproducibility Analysis by Country"
        Main title for the figure
    fig_size : tuple, default=(15, 10)
        Figure size
        
    Returns:
    --------
    fig : matplotlib Figure object
    """
    # Filter data
    filtered_df = df[df['Major claims'] >= min_claims].copy()
    
    # Aggregate data by country
    categories = ['Challenged', 'Mixed', 'Unchallenged', 'Partially Verified', 'Verified']
    
    # Group by country
    country_sums = filtered_df.groupby(country_col)[categories + ['Major claims', 'Articles']].sum().reset_index()
    
    # Calculate proportions
    for cat in categories:
        country_sums[f'{cat}_prop'] = country_sums[cat] / country_sums['Major claims']
    
    # Sort countries based on the specified criterion
    if sort_by in country_sums.columns:
        country_sums = country_sums.sort_values(sort_by, ascending=False)
    
    # Get list of countries in proper order
    countries = country_sums[country_col].tolist()
    
    # Adjust figure height based on number of countries
    country_count = len(countries)
    if country_count < 5:
        fig_size = (fig_size[0], max(4, country_count * 0.8))
    elif country_count > 12:
        fig_size = (fig_size[0], min(16, country_count * 0.6))
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size, gridspec_kw={'width_ratios': [3, 1]})
    
    # Panel 1: Stacked proportions
    y_pos = np.arange(len(countries))
    lefts = np.zeros(len(countries))
    
    # Create stacked bars
    for category in ASSESSMENT_ORDER:
        prop_col = f'{category}_prop'
        widths = country_sums[prop_col].values
        
        # Plot this category as a segment in the stacked bar
        bar = ax1.barh(
            y_pos,
            widths,
            left=lefts,
            color=ASSESSMENT_COLORS[category],
            label=category,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.9
        )
        
        # Update left positions for next category
        lefts += widths
        
        # Add text labels to segments that are large enough
        for i, width in enumerate(widths):
            if width > 0.05:  # Only label segments that are at least 5%
                # Position text in center of segment
                x_pos = lefts[i] - width/2
                
                # Add percentage label
                ax1.text(
                    x_pos,
                    y_pos[i],
                    f'{width:.0%}' if width >= 0.1 else f'{width:.1%}',
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold',
                    fontsize=11
                )
    
    # Set country names on y-axis
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(countries)
    
    # Set axis labels and title
    ax1.set_xlabel('Proportion of Claims', fontweight='bold')
    ax1.set_ylabel('Country', fontweight='bold')
    ax1.set_title('Assessment Categories by Country', fontweight='bold')
    
    # Format x-axis as percentage
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Set x-axis limit
    ax1.set_xlim(0, 1.0)
    
    # Add grid
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Number of claims bar chart
    bars = ax2.barh(
        y_pos,
        country_sums['Major claims'],
        color='#3498db',  # Blue
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(
            width/2,
            bar.get_y() + bar.get_height()/2,
            f"{int(width)}",
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
            fontsize=11
        )
    
    # Set axis labels and title
    ax2.set_xlabel('Number of Claims', fontweight='bold')
    ax2.set_title('Total Claims by Country', fontweight='bold')
    
    # Hide y-axis labels (shared with first panel)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    
    # Add grid
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add legend below the figure
    handles = [Patch(facecolor=ASSESSMENT_COLORS[cat], edgecolor='white', label=cat) for cat in ASSESSMENT_ORDER]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(ASSESSMENT_ORDER),
        frameon=True,
        framealpha=0.9,
        edgecolor='lightgray',
        title="Assessment Category",
        bbox_to_anchor=(0.5, 0.02)  # FIXED: Adjusted position
    )
    
    # Add main title
    fig.suptitle(title, fontsize=BIGGER_SIZE+2, fontweight='bold', y=0.98)
    
    # FIXED: Better layout adjustment
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.90, wspace=0.05)
    
    return fig, (ax1, ax2)


# Example usage for two-panel chart:
fig, (ax1, ax2) = create_two_panel_country_chart(
    df=to_plot,
    min_claims=5,
    sort_by='Challenged_prop',
    title="Scientific Reproducibility Analysis by Country"
)
# TODO: sort then on opartially verified.
plt.savefig('figures/fig11C_country.png', dpi=300, bbox_inches='tight')

# %%
major_claims_df["year"].max()

# %%


# %%


# %%
to_plot

# %%


# %%
# Create a copy to avoid modifying the original
ranking_df = major_claims_df.copy()
# 
# # Use numpy.select to create categories based on conditions
# conditions = [
#     (ranking_df['shangai_ranking_2010'] <= 50) & (~pd.isna(ranking_df['shangai_ranking_2010'])),
#     (ranking_df['shangai_ranking_2010'] > 50) & (ranking_df['shangai_ranking_2010'] <= 100),
#     (ranking_df['shangai_ranking_2010'] > 100)  & (~pd.isna(ranking_df['shangai_ranking_2010']))
# ]
# 
# choices = ['Top 50', '51-100', '101+']
# 
# # Create ranking category column, defaulting to "Not Ranked" for NA values
# ranking_df['ranking_category'] = np.select(conditions, choices, default='Not Ranked')

variable = "ranking_category"
var_grouped = wrangling.create_author_metric(claim_df = ranking_df, 
                                        variable= variable,
                                        other_col={"n_university":('primary_affiliation','nunique')}).set_index(variable)

for col in assessment_columns:
    var_grouped[f'{col}_prop'] = var_grouped[col] / var_grouped['Major claims']

print(f"Summary of {variable}:")
print(var_grouped[['Major claims', 'Articles', 
                    'Verified_prop', 'Challenged_prop', 'Unchallenged_prop']])
print(var_grouped.index)

labels_map = {
    'Top 50': 'Top 50',
    '51-100': '51-100',
    '101+': '101+',
    'Not Ranked': 'Not Ranked'
}



fig, ax = plot_info.create_horizontal_bar_chart(var_grouped, 
                                                show_p_value=False, 
                                                labels_map=labels_map, 
                                                title="Shangai ranking",
                                                other_n={"Institutions": "n_university"})
plt.savefig(f"figures/fig2C_Shangai_ranking.png", dpi=300, bbox_inches='tight')

# %%


# %%
# ── Figure 2 composite ──────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plot_info

fig = plt.figure(figsize=(16, 11))

# GridSpec: 2 rows × 2 cols  (heights 30 %, 70 %; widths 50 %, 50 %)
gs = gridspec.GridSpec(
    2, 2,
    width_ratios=[1, 1],
    height_ratios=[0.5, 0.5],
    wspace=0.15,
    hspace=0.15
)

axA = fig.add_subplot(gs[0, 0])     # Panel A
axB = fig.add_subplot(gs[1, 0])     # Panel B
axC = fig.add_subplot(gs[:, 1])     # Panel C (spans both rows)

# ── Panel A – absolute counts ─────────────────────────────────────────
plot_info.create_stacked_bar_plot(
    major_claims_df,
    mode='absolute',
    by_time=False,
    use_expanded=False,
    ax=axA
)
axA.set_title("A", loc='left', fontweight='bold', fontsize=30)
axA.set_xlabel("")
axA.set_xticklabels([])

# ── Panel B – percentage ──────────────────────────────────────────────
plot_info.create_stacked_bar_plot(
    major_claims_df,
    mode='percentage',
    by_time=False,
    use_expanded=False,
    ax=axB
)
axB.set_title("B", loc='left', fontweight='bold', fontsize=30)

# ── Panel C – institution bar chart (vertical) ────────────────────────

#labels_map = {
#         'Top 50': 'Top 50',
#     'Not Ranked': 'Not Ranked',
#         '101+': '101+',
#         '51-100': '51-100',
#
# }
plot_info.create_horizontal_bar_chart(
    var_grouped,
    title="",
    labels_map=labels_map,
    show_p_value=False,
    other_n={"Institutions": "n_university"},
    ax=axC,
    orientation="vertical",
    pct_axis_label="% of Claims",        # controls x (horizontal) or y (vertical)
    group_axis_label="Institution Shanghai ranking"  # controls y (horizontal) or x (vertical)
)
axC.set_title("C", loc='left', fontweight='bold', fontsize=30)

# remove legends from B and C
for ax in (axB, axC):
    leg = ax.get_legend()
    if leg:
        leg.remove()

# --- grab legend entries from panel B ---------------------------------
legA = axA.get_legend()           # <-- this should exist after the plot call
handles = legA.legend_handles
labels  = [t.get_text() for t in legA.get_texts()]
# unified legend, anchored inside Panel A, upper-right
axA.legend(
    handles,
    labels,
    #title="Assessment Category",
    loc="upper right",       # 1 = upper-right corner of the Axes
    bbox_to_anchor=(1.0, 1.0),   # (x,y) with Axes coords
    ncol=1,                 # single column fits nicely
    frameon=True,
    fontsize=18,
)
fig.tight_layout(rect=[0, 0.07, 1, 1])  # leave space at bottom for legend
fig.savefig('figures/fig2_ABC_claims_distribution.png',
            dpi=300, bbox_inches='tight')
print("Saved → figures/fig2_ABC_claims_distribution.png")



# %%
# ── Figure 3 composite: time‑period claims distribution (A–B) ─────────────
# Two vertical panels: A = absolute counts ─ time periods
#                      B = percentage ─ time periods

fig3 = plt.figure(figsize=(8, 13))
gs3  = gridspec.GridSpec(
    2, 1,
    height_ratios=[0.5, 0.5],
    hspace=0.20
)

ax3A = fig3.add_subplot(gs3[0])
ax3B = fig3.add_subplot(gs3[1])

# Panel A – absolute counts across time periods
plot_info.create_stacked_bar_plot(
    major_claims_df,
    mode="absolute",
    by_time=True,
    use_expanded=False,
    ax=ax3A,
)
ax3A.set_title("A", loc="left", fontweight="bold", fontsize=30)
ax3A.set_xlabel("")           # remove redundant x‑label
ax3A.set_xticklabels([])

# Panel B – percentage across time periods
plot_info.create_stacked_bar_plot(
    major_claims_df,
    mode="percentage",
    by_time=True,
    use_expanded=False,
    ax=ax3B,
)
ax3B.set_title("B", loc="left", fontweight="bold", fontsize=30)

leg = ax3B.get_legend()
if leg:
    leg.remove()

# Unify legend: use the one generated in panel A (it has all categories)
leg3 = ax3A.get_legend()
handles3 = leg3.legend_handles
labels3  = [t.get_text() for t in leg3.get_texts()]
leg3.remove()  # remove default legend from A

ax3A.legend(
    handles3,
    labels3,
    loc="upper left",
    frameon=True,
    ncol=1,
    fontsize=18,
)

fig3.tight_layout(rect=[0, 0.02, 1, 1])
fig3.savefig("figures/fig3_AB_time_period.png", dpi=300, bbox_inches="tight")
print("Saved → figures/fig3_AB_time_period.png")

# %%


# %%

to_plot = wrangling.create_author_metric(
    claim_df=major_claims_df, 
    variable='primary_affiliation', 
    other_col={"country":('country', 'first'), 
               "shangai_ranking_2010":('shangai_ranking_2010', 'first')})




to_plot = to_plot[~pd.isna(to_plot['shangai_ranking_2010'])]
fig, ax = plot_info.create_publication_scatter(
    to_plot,
    x_var='shangai_ranking_2010', 
    y_var='Challenged prop',
    size_var='Articles', 
    title='Challenged vs. Unchallenged Claims by Author',
    x_label='Proportion of Unchallenged Claims',
    y_label='Proportion of Challenged Claims',
    show_regression=False,
    annotate_top_n=0
)

# %%
# Import stat_lib functions for statistical reporting
import stat_lib

# %%
# Journal proportions and statistical comparisons
print("=== JOURNAL CATEGORY PROPORTIONS AND COMPARISONS ===\n")

# Calculate proportions for each journal category (using plot order)
journal_categories = ['Low Impact', 'High Impact', 'Trophy Journals']  # From plot_info.py line 245
print("1. PROPORTIONS BY JOURNAL CATEGORY:")
print("-" * 50)

for category in journal_categories:
    category_data = major_claims_df[major_claims_df['journal_category'] == category]
    challenged_count = len(category_data[category_data['assessment_type_grouped'] == 'Challenged'])
    total_count = len(category_data)
    
    proportion_report = stat_lib.report_proportion(
        successes=challenged_count,
        total=total_count,
        end_sentence=f"of {category} claims were challenged."
    )
    print(f"{category}: {proportion_report}")

print("\n2. STATISTICAL COMPARISONS (using Low-Impact as baseline):")
print("-" * 60)

# Create summary data for comparisons
category_summary = major_claims_df.groupby('journal_category').agg({
    'assessment_type_grouped': lambda x: (x == 'Challenged').sum(),  # Count of challenged
    'journal_category': 'count'  # Total count
}).rename(columns={'assessment_type_grouped': 'Challenged', 'journal_category': 'Major claims'})

# High-Impact vs Low-Impact comparison
if 'High Impact' in category_summary.index and 'Low Impact' in category_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=category_summary,
        labels=['High Impact', 'Low Impact'],
        outcome='Challenged',
        what_str="High-impact vs low-impact journal category"
    )
    print("High-Impact vs Low-Impact:")
    print(comparison_sentence)
    print()

# Trophy vs Low-Impact comparison
if 'Trophy Journals' in category_summary.index and 'Low Impact' in category_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=category_summary,
        labels=['Trophy Journals', 'Low Impact'],
        outcome='Challenged',
        what_str="Trophy vs low-impact journal category"
    )
    print("Trophy vs Low-Impact:")
    print(comparison_sentence)
    print()

# Trophy vs High-Impact comparison  
if 'Trophy Journals' in category_summary.index and 'High Impact' in category_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=category_summary,
        labels=['Trophy Journals', 'High Impact'],
        outcome='Challenged',
        what_str="Trophy vs high-impact journal category"
    )
    print("Trophy vs High-Impact:")
    print(comparison_sentence)

# %%
# UNCHALLENGED CLAIMS ANALYSIS
print("\n=== UNCHALLENGED CLAIMS PROPORTIONS AND COMPARISONS ===\n")

# Calculate proportions for unchallenged claims by journal category
print("1. UNCHALLENGED PROPORTIONS BY JOURNAL CATEGORY:")
print("-" * 50)

for category in journal_categories:  # Using same order as above
    category_data = major_claims_df[major_claims_df['journal_category'] == category]
    unchallenged_count = len(category_data[category_data['assessment_type_grouped'] == 'Unchallenged'])
    total_count = len(category_data)
    
    proportion_report = stat_lib.report_proportion(
        successes=unchallenged_count,
        total=total_count,
        end_sentence=f"of {category} claims were unchallenged."
    )
    print(f"{category}: {proportion_report}")

print("\n2. UNCHALLENGED STATISTICAL COMPARISONS (using Low-Impact as baseline):")
print("-" * 70)

# Create summary data for unchallenged comparisons
unchallenged_summary = major_claims_df.groupby('journal_category').agg({
    'assessment_type_grouped': lambda x: (x == 'Unchallenged').sum(),  # Count of unchallenged
    'journal_category': 'count'  # Total count
}).rename(columns={'assessment_type_grouped': 'Unchallenged', 'journal_category': 'Major claims'})

# High-Impact vs Low-Impact comparison for unchallenged
if 'High Impact' in unchallenged_summary.index and 'Low Impact' in unchallenged_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=unchallenged_summary,
        labels=['High Impact', 'Low Impact'],
        outcome='Unchallenged',
        what_str="High-impact vs low-impact journal category"
    )
    print("High-Impact vs Low-Impact (Unchallenged):")
    print(comparison_sentence)
    print()

# Trophy vs Low-Impact comparison for unchallenged
if 'Trophy Journals' in unchallenged_summary.index and 'Low Impact' in unchallenged_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=unchallenged_summary,
        labels=['Trophy Journals', 'Low Impact'],
        outcome='Unchallenged',
        what_str="Trophy vs low-impact journal category"
    )
    print("Trophy vs Low-Impact (Unchallenged):")
    print(comparison_sentence)
    print()

# Trophy vs High-Impact comparison for unchallenged
if 'Trophy Journals' in unchallenged_summary.index and 'High Impact' in unchallenged_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=unchallenged_summary,
        labels=['Trophy Journals', 'High Impact'],
        outcome='Unchallenged',
        what_str="Trophy vs high-impact journal category"
    )
    print("Trophy vs High-Impact (Unchallenged):")
    print(comparison_sentence)

# %%
# UNIVERSITY RANKING ANALYSIS - CHALLENGED CLAIMS
print("\n=== UNIVERSITY RANKING CHALLENGED CLAIMS ANALYSIS ===\n")

# Calculate proportions for challenged claims by university ranking (using plot order)
university_rankings = ['Top 50', '51-100', '101+', 'Not Ranked']  # From labels_map order in plotting
print("1. CHALLENGED PROPORTIONS BY UNIVERSITY RANKING:")
print("-" * 55)

for ranking in university_rankings:
    ranking_data = major_claims_df[major_claims_df['ranking_category'] == ranking]
    challenged_count = len(ranking_data[ranking_data['assessment_type_grouped'] == 'Challenged'])
    total_count = len(ranking_data)
    
    proportion_report = stat_lib.report_proportion(
        successes=challenged_count,
        total=total_count,
        end_sentence=f"of {ranking} university claims were challenged."
    )
    print(f"{ranking}: {proportion_report}")

print("\n2. UNIVERSITY RANKING STATISTICAL COMPARISONS (using Not Ranked as baseline):")
print("-" * 80)

# Create summary data for university ranking comparisons
university_summary = major_claims_df.groupby('ranking_category').agg({
    'assessment_type_grouped': lambda x: (x == 'Challenged').sum(),  # Count of challenged
    'ranking_category': 'count'  # Total count
}).rename(columns={'assessment_type_grouped': 'Challenged', 'ranking_category': 'Major claims'})

# Top 50 vs Not Ranked comparison
if 'Top 50' in university_summary.index and 'Not Ranked' in university_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=university_summary,
        labels=['Top 50', 'Not Ranked'],
        outcome='Challenged',
        what_str="Top 50 vs not ranked university"
    )
    print("Top 50 vs Not Ranked:")
    print(comparison_sentence)
    print()

# 51-100 vs Not Ranked comparison
if '51-100' in university_summary.index and 'Not Ranked' in university_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=university_summary,
        labels=['51-100', 'Not Ranked'],
        outcome='Challenged',
        what_str="51-100 ranked vs not ranked university"
    )
    print("51-100 vs Not Ranked:")
    print(comparison_sentence)
    print()

# 101+ vs Not Ranked comparison
if '101+' in university_summary.index and 'Not Ranked' in university_summary.index:
    comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
        var_grouped=university_summary,
        labels=['101+', 'Not Ranked'],
        outcome='Challenged',
        what_str="101+ ranked vs not ranked university"
    )
    print("101+ vs Not Ranked:")
    print(comparison_sentence)

# %%
# TIME PERIOD ANALYSIS - CHALLENGED CLAIMS
print("\n=== TIME PERIOD CHALLENGED CLAIMS ANALYSIS ===\n")

# Calculate proportions for challenged claims by time period (using plot order)
time_periods = ['≤1991', '1992-1996', '1997-2001', '2002-2006', '2007-2011']  # From plot_info.py line 240
print("1. CHALLENGED PROPORTIONS BY TIME PERIOD:")
print("-" * 45)

for period in time_periods:
    period_data = major_claims_df[major_claims_df['year_binned'] == period]
    challenged_count = len(period_data[period_data['assessment_type_grouped'] == 'Challenged'])
    total_count = len(period_data)
    
    proportion_report = stat_lib.report_proportion(
        successes=challenged_count,
        total=total_count,
        end_sentence=f"of {period} claims were challenged."
    )
    print(f"{period}: {proportion_report}")

print("\n2. TIME PERIOD STATISTICAL COMPARISONS (pairwise):")
print("-" * 55)

# Create summary data for time period comparisons
time_summary = major_claims_df.groupby('year_binned').agg({
    'assessment_type_grouped': lambda x: (x == 'Challenged').sum(),  # Count of challenged
    'year_binned': 'count'  # Total count
}).rename(columns={'assessment_type_grouped': 'Challenged', 'year_binned': 'Major claims'})

# Use predefined time periods in chronological order
sorted_periods = time_periods  # Already in chronological order

# Compare consecutive time periods
for i in range(len(sorted_periods) - 1):
    period1 = sorted_periods[i]
    period2 = sorted_periods[i + 1]
    
    if period1 in time_summary.index and period2 in time_summary.index:
        comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
            var_grouped=time_summary,
            labels=[period2, period1],  # Later period vs earlier period
            outcome='Challenged',
            what_str=f"{period2} vs {period1} time period"
        )
        print(f"{period2} vs {period1}:")
        print(comparison_sentence)
        print()

# Also compare lowerst vs most period
if len(sorted_periods) >= 2:
    print("ok")
    earliest = sorted_periods[0]
    latest = sorted_periods[-2] # minus one is idex of the lowest
    
    if earliest != latest and earliest in time_summary.index and latest in time_summary.index:
        comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
            var_grouped=time_summary,
            labels=[latest, earliest],
            outcome='Challenged',
            what_str=f"{latest} vs {earliest} time period"
        )
        print(f"{latest} vs {earliest} (Overall trend):")
        print(comparison_sentence)

# %%



