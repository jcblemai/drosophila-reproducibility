# %% [markdown]
# # Figure 4: First authors

# %%
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

import plot_info # Does this set the plot style ? Yes it does
import wrangling
import stat_lib

assessment_columns = plot_info.assessment_columns

first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")

author_metrics = wrangling.create_author_metric(claim_df=first_author_claims, 
                                                variable='first_author_key', other_col={"First Author Name":('First Author Name', 'first')})

# %%


# %%
descriptive_stats = author_metrics[assessment_columns + [col + ' prop' for col in assessment_columns] + 
                    ['Challenged prop', 'Articles', 'Major claims']].describe()

sns.histplot(author_metrics['Challenged prop'], kde=True)
plt.title('Distribution of % of challenged claims')
plt.xlabel('Proportion of Challenged Claims')
plt.axvline(author_metrics['Challenged prop'].mean(), color='red', linestyle='--', 
            label=f'Mean = {author_metrics["Challenged prop"].mean():.2f}')
plt.legend()
plt.show()
print("Descriptive Statistics:")
descriptive_stats

# %% [markdown]
# ## A. Distribution plot

# %%
to_plot = author_metrics.copy()
to_plot = to_plot[to_plot['Articles'] >= 0]
to_plot = to_plot[to_plot['Major claims'] >= 0]

importlib.reload(plot_info)
fig, ax = plot_info.plot_author_irreproducibility_focused(
    df=to_plot,
    title="",
    color_by='Verified prop',
    cmap='RdYlGn',  # Choose colormap that doesn't have white at minimum
    most_challenged_on_right=True,
    name_col='First Author Name',
)
plt.savefig('figures/fig4A_distribution_scatter.png', dpi=300, bbox_inches='tight')
importlib.reload(plot_info)
fig1, ax1 = plot_info.plot_challenged_histogram(to_plot, title="",)
#plt.savefig('figures/fig4A-V2.png', dpi=300, bbox_inches='tight')

# Create Lorenz curve visualization
importlib.reload(plot_info)
fig2, ax2 = plot_info.plot_lorenz_curve(to_plot, title="",)
plt.savefig('figures/fig4B_distribution_gini.png', dpi=300, bbox_inches='tight')

# %%
 

# %% [markdown]
# ## B. Binary variables

# %%
first_author_claims["First Author Career Stage"].unique()

# %%
all_categorical_variables = {
            # "PhD Post-doc": {
            #     "labels": ['PhD', 'Post-doc'],
            #     "title": "Assessment of Scientific Claims by Career Stage",
            #     "fig_name_prefix" : "fig5B-Vb_"
            #     },
            # "Is a PI": {
            #     "labels": ['Not PI', 'PI'],
            #     "title": "Assessment of Scientific Claims by Career Stage",
            #     "fig_name_prefix" : "fig5B-Va_"
            #     },
            "First Author Career Stage": {
                "labels": {v:v for v in first_author_claims["First Author Career Stage"].unique()},
                "title": "Assessment of Scientific Claims by Career Stage",
                "fig_name_prefix" : "fig5B_"
                },
            "First Author Sex": {
                "labels": None, 
                "title": "Assessment of Scientific Claims by Sex",
                "fig_name_prefix" : "fig5A_"
            },
            #"MD": {
            #    "labels": ["Not-MD", "MD"],
            #    "title": "Assessment of Scientific Claims by MD Degree",
            #    "fig_name_prefix" : "fig5X_"
            #},
            "First Author Become a PI": {
                "labels": {
                    False:"First author that\ndo not become PIs", 
                    True:"First authors that\nbecome PIs"
                },
                #"title": "Assessment of Scientific Claims for first authors that become PIs",
                "title": "",
                "fig_name_prefix" : "fig10_"
            }
}

all_comparisons = []

for variable in all_categorical_variables.keys():
    # This was wrong because it overcounted claims when an author had different values for the same variable Utiliser pour les 2025-04-09
    # var_grouped = plot_info.prepare_categorical_variable_data(first_author_claims, author_metrics, variable, 
    #                                                 key_col='first_author_key',
    #                                                 assessment_columns=assessment_columns)
    
    var_grouped = wrangling.create_author_metric(claim_df = first_author_claims, 
                                        variable= variable,
                                        other_col={"Name":('First Author Name', 'first'), 
                                                "n_authors":('first_author_key', 'nunique')}).set_index(variable)

    labels = all_categorical_variables[variable]["labels"]
    #if set(var_grouped.index) == set(labels):
    actual_groups = list(var_grouped.index)
    # Print to check if needed:
    print("Actual groups found:", actual_groups)
    if len(actual_groups) == 2:
        sentence, summary = stat_lib.report_categorical_comparison(var_grouped, 
                                                                        actual_groups, 
                                                                        outcome='Challenged',
                                                                        what_str=f"First Author {variable} ")
        print("\n"+sentence+"\n")
        all_comparisons.append({'Variable': variable, **summary})
    
    explain_df = first_author_claims.groupby(["First Author Name", variable]).agg(**{
        "Major claims":('id', 'count'),
        "Articles":('article_id', 'nunique')
        },
    ).reset_index().pivot(index="First Author Name", columns=variable, values="Major claims")
    explain_df.to_csv(f"figures/figure_tables/{all_categorical_variables[variable]['fig_name_prefix']}categorical_{variable}.csv", 
                    index=True, index_label=f"Name/{variable}", sep=";")


    # Calculate proportions
    for col in assessment_columns:
        var_grouped[f'{col}_prop'] = var_grouped[col] / var_grouped['Major claims']

    print(f"Summary of {variable}:")
    print(var_grouped[['Major claims', 'Articles', 
                        'Verified_prop', 'Challenged_prop', 'Unchallenged_prop']])

    
    labels=all_categorical_variables[variable]["labels"]

    importlib.reload(plot_info)
    fig, ax = plot_info.create_horizontal_bar_chart(var_grouped, 
                                                    show_p_value=False, 
                                                    labels_map=labels, 
                                                    title=all_categorical_variables[variable]["title"],
                                                    other_n={"authors": "n_authors"})
    plt.savefig(f"figures/{all_categorical_variables[variable]['fig_name_prefix']}categorical_{variable}.png", dpi=300, bbox_inches='tight')

comparison_df = pd.DataFrame(all_comparisons)
comparison_df.to_csv("figures/figure_tables/summary_categorical_comparisons_first.csv", sep=";", index=False)

# %%


# %%
author_metrics

# %%
to_plot = author_metrics.copy()

# %%

# Create article count bins
article_bins = [.5, 1.5, 2.5, 3.5, 4.5,  float('inf')]
bin_labels =     ['1', '2', '3', '4', '5']
to_plot['Article_bin'] = pd.cut(to_plot['Articles'], bins=article_bins, labels=bin_labels)

# Create the boxplot
ax = sns.boxplot(x='Article_bin', y='Challenged prop', data=to_plot)
#ax = sns.stripplot(x='Article_bin', y='Challenged prop', data=to_plot, alpha=0.3, jitter=True, jitter_size=.95, color='black')

# Add individual data points
#sns.stripplot(x='Article_bi  n', y='Challenged prop', data=to_plot, 
#              color='black', alpha=0.3, jitter=True)

# Add count annotations below each category
for i, cat in enumerate(bin_labels):
    count = len(to_plot[to_plot['Article_bin'] == cat])
    ax.text(i, 1.05, f"n={count}", ha='center', va='top', fontsize=12,
            transform=ax.get_xaxis_transform())

# Format the y-axis as percentage
ax.yaxis.set_major_formatter(plot_info.PercentFormatter(1.0))

# Add labels and title
plt.xlabel('Number of Articles', fontsize=14, fontweight='bold')
plt.ylabel('Proportion of Challenged Claims', fontsize=14, fontweight='bold')
plt.title('Proportion of Challenged Claims by Author Publication Volume', 
          fontsize=16, fontweight='bold', pad=20)

# Add median values on top of each box
medians = to_plot.groupby('Article_bin')['Challenged prop'].median()
for i, m in enumerate(medians):
    ax.text(i, m + 0.01, f"{m:.1%}", ha='center', va='bottom', fontweight='bold')

# Remove top and right spines
sns.despine()

plt.tight_layout()
# plt.savefig('figures/fig4B-nb_article_V1.png', dpi=300, bbox_inches='tight')

# %%
importlib.reload(plot_info)
fig1, ax1 = plot_info.create_challenged_vs_unchallenged_scatter(to_plot,size_mult=100, name_col='First Author Name',)

# %%

importlib.reload(plot_info)
fig2, ax2 = plot_info.create_challenged_vs_articles_scatter(to_plot, name_col='First Author Name')
plt.savefig(f'figures/fig5C_scatterA.png', dpi=300, bbox_inches='tight')



# %%
to_plot["Major claims"].value_counts().sort_index().plot(kind='bar')    
to_plot["Articles"].value_counts().sort_index().plot(kind='bar')    

# %% [markdown]
# ### Continuous Variables

# %%


# %%


# %%
importlib.reload(plot_info)
fig, ax = plot_info.create_publication_scatter(
    to_plot,
    x_var='Unchallenged prop', 
    y_var='Challenged prop',
    size_var='Articles', 
    title='Challenged vs. Unchallenged Claims by Author',
    x_label='Proportion of Unchallenged Claims',
    y_label='Proportion of Challenged Claims',
    annotate_top_n=5, name_col='First Author Name'
)
#plt.savefig(f'figures/fig4C_scatterB', dpi=300, bbox_inches='tight')

# %% [markdown]
# ### Figure 4 – Gini (A) and Distribution (B) side-by-side

# %%
import matplotlib.gridspec as gridspec
from plot_info import MEDIUM_SIZE

to_plot = author_metrics.copy()
# to_plot = to_plot[to_plot['Articles'] >= 2]
# to_plot = to_plot[to_plot['Major claims'] >= 6]

fig4 = plt.figure(figsize=plot_info.HORIZONTAL_LAYOUT)
gs4  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)

ax4A = fig4.add_subplot(gs4[0])   # A – Lorenz / Gini
ax4B = fig4.add_subplot(gs4[1])   # B – scatter

# Panel A: Lorenz curve + Gini
importlib.reload(plot_info)
plot_info.plot_lorenz_curve(
    to_plot,
    prop_column='Challenged prop',
    weight_column='Major claims',
    title="",
    ax=ax4A,
    print_gini=True,
    print_top_txt=False
)
ax4A.set_title("A", loc='left', fontweight='bold', fontsize=plot_info.PANEL_LABEL_SIZE)

# Panel B: distribution scatter
importlib.reload(plot_info)
plot_info.plot_author_irreproducibility_focused(
    to_plot,
    title="",
    color_by='Verified prop',
    cmap='RdYlGn',
    most_challenged_on_right=True,
    annotate_top_n=0,
    name_col='First Author Name',
    ax=ax4B
)
ax4B.set_title("B", loc='left', fontweight='bold', fontsize=plot_info.PANEL_LABEL_SIZE)

fig4.tight_layout()
fig4.savefig("figures/fig4_AB_first_author_horizontal.png",
             dpi=300, bbox_inches="tight")
print("Saved → figures/fig4_AB_first_author_horizontal.png")


# %%
# %% [markdown]
# ### Figure 5 – Author characteristics (A–C)

# %%
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------
# Composite Figure 5 layout:
#   ┌───────────────┬───────────────────────────────┐
#   │       A       │               B               │   25 % / 75 % width
#   ├───────────────┴───────────────────────────────┤
#   │                     C                         │   full width
#   └───────────────────────────────────────────────┘

fig5 = plt.figure(figsize=plot_info.COMPLEX_LAYOUT)
gs5  = gridspec.GridSpec(
    2, 2,
    width_ratios=[0.25, 0.75],
    height_ratios=[0.5, 0.5],
    wspace=0.25,
    hspace=0.5
)

# Panel‑A axis (top‑left)
axA = fig5.add_subplot(gs5[0, 0])

# Panel‑B axis (top‑right), share y with A
axB = fig5.add_subplot(gs5[0, 1], sharey=axA)

# Panel‑C axis (bottom, spans both columns)
axC = fig5.add_subplot(gs5[1, :])

# ------------------------------------------------------------------
# Panel A – First Author Sex  (left, width 0.2)
varA        = "First Author Sex"
label_mapA  = None

grpA = wrangling.create_author_metric(
    claim_df=first_author_claims,
    variable=varA,
    other_col={"n_authors": ('first_author_key', 'nunique')}
).set_index(varA)

for col in plot_info.assessment_columns:
    grpA[f'{col}_prop'] = grpA[col] / grpA['Major claims']

importlib.reload(plot_info)
plot_info.create_horizontal_bar_chart(
    grpA,
    title="",
    labels_map=label_mapA,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    orientation="vertical",
    pct_axis_label="% of Claims",
    group_axis_label=varA,
    ax=axA,
)
axA.set_title("A", loc="left", fontweight="bold", fontsize=plot_info.PANEL_LABEL_SIZE, x=-0.2, y=1.05)

# Save legend handles from panel A
lg = axA.get_legend()
legend_handles = lg.legend_handles
legend_labels  = [t.get_text() for t in lg.get_texts()]
lg.remove()

# ------------------------------------------------------------------
# Panel B – First Author Career Stage  (middle, width 0.4)
varB        = "First Author Career Stage"
label_mapB  = all_categorical_variables[varB]["labels"]

grpB = wrangling.create_author_metric(
    claim_df=first_author_claims,
    variable=varB,
    other_col={"n_authors": ('first_author_key', 'nunique')}
).set_index(varB)

for col in plot_info.assessment_columns:
    grpB[f'{col}_prop'] = grpB[col] / grpB['Major claims']

importlib.reload(plot_info)
plot_info.create_horizontal_bar_chart(
    grpB,
    title="",
    labels_map=label_mapB,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    orientation="vertical",
    pct_axis_label="% of Claims",
    group_axis_label=varB,
    ax=axB,
)
axB.set_title("B", loc="left", fontweight="bold", fontsize=plot_info.PANEL_LABEL_SIZE, x=-0.1, y=1.05)
# Ensure panel B legend is removed
if axB.get_legend():
    axB.get_legend().remove()

# ── Panel C : scatter – challenged vs. articles ───────────────────

to_plot = author_metrics.copy()
importlib.reload(plot_info)
plot_info.create_challenged_vs_articles_scatter(
    to_plot,
    annotate_top_n=0,
    title="",
    size_mult=100,
    #name_col='First Author Name',
    ax=axC
)
axC.set_title("C", loc="left", fontweight="bold", fontsize=plot_info.PANEL_LABEL_SIZE, x=-0.1, y=1.05)

# Unified legend in upper-right of panel A
fig5.axes[1].legend(
    legend_handles,
    legend_labels,
    loc="upper left",
    frameon=True,
    ncol=1,
    fontsize=MEDIUM_SIZE,
)

fig5.tight_layout()
fig5.savefig("figures/fig5_ABC_mixed_panels_horizontal.png",
             dpi=300, bbox_inches="tight")
print("Saved → figures/fig5_ABC_mixed_panels_horizontal.png")
# %%
