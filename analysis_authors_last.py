# %% [markdown]
# # Figure 5: Leading authors

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import stat_lib

from pathlib import Path
# Create directory structure if it doesn't exist
Path(f"figures/figure_tables").mkdir(parents=True, exist_ok=True)


import plot_info # Does this set the plot style ? Yes it does
import wrangling

assessment_columns = plot_info.assessment_columns

leading_author_claims = pd.read_csv("preprocessed_data/leading_author_claims.csv")
author_metrics = wrangling.create_author_metric(claim_df = leading_author_claims,
                                                variable= "leading_author_key",
                                                other_col={"Leading Author Name":('Leading Author Name', 'first')},)
author_metrics_after_2000 = wrangling.create_author_metric(
                                                claim_df=leading_author_claims[leading_author_claims['year'] > 2000], 
                                                variable= "leading_author_key", 
                                                other_col={"Leading Author Name":('Leading Author Name', 'first')},)


# %%
author_metrics

# %%
leading_author_claims.columns

# %%
print(len(author_metrics))
# TODO update all with this ?
print(len(author_metrics))

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
min_articles = 0
min_claims = 0
to_plot = to_plot[to_plot['Articles'] >= min_articles]
to_plot = to_plot[to_plot['Major claims'] >= min_claims]
print(f"Our analysis will focus on the {len(to_plot)} PIs who published at least {min_claims} major claims from a"
      f"minimum of {min_articles} articles")
print(f"Among the {len(to_plot)} PIs, we identified {len(to_plot[to_plot['Challenged prop']>.1])} laboratories "
      f"where more than 10% of the claims were challenged, with the highest proportion reaching"
      f" {max(to_plot['Challenged prop']*100)}% ")


fig, ax = plot_info.plot_author_irreproducibility_focused(
    df=to_plot,
    title="",
    color_by='Verified prop',
    cmap='RdYlGn',  # Choose colormap that doesn't have white at minimum
    most_challenged_on_right=True,
    name_col='Leading Author Name',
)
plt.savefig('figures/fig6A_distribution_scatter.png', dpi=300, bbox_inches='tight')
fig1, ax1 = plot_info.plot_challenged_histogram(to_plot,title="",)
#plt.savefig('figures/fig5A-V2.png', dpi=300, bbox_inches='tight')

# Create Lorenz curve visualization
fig2, ax2 = plot_info.plot_lorenz_curve(to_plot,
                                        title="",)
plt.savefig('figures/fig6B_distribution_gini.png', dpi=300, bbox_inches='tight')

# %%
author_metrics

# %%


# %%


# %% [markdown]
# ## B. Binary variables

# %%
all_categorical_variables = {
            "Historical lab after 1998": {
                "labels": {
                    False:'PI not trained in\ntraditional laboratories', 
                    True:'PI trained in\ntraditional laboratories'},
                "title": "Assessment of Scientific Claims by Laboratory Tradition (only articles after 1995)",
                "fig_name_prefix" : "fig8C_"
                },
            "Continuity": {
                "labels": {
                    False:'Exploratory PI', 
                    True:'Continuity PI'},
                "title": "Assessment of Scientific Claims by Continuity",
                "fig_name_prefix" : "fig9A_"
                },
            "Leading Author Sex": {
                "labels" : None,
                "title": "Assessment of Scientific Claims by Sex",
                "fig_name_prefix" : "fig7A_",
                },
            "Junior Senior": {
                "labels": None,
                "title": "Assessment of Scientific Claims by Junior/Senior PI",
                "fig_name_prefix" : "fig7C_"
                },
            "F and L": {
                "labels": {
                    False:"PI that were\nnot first author", 
                    True:"PI that were\nfirst author"},
                "title": "Assessment of Scientific Claims by previous mentee experience \n (Only author that published at least once after 1998, counting after 1995)",
                "fig_name_prefix" : "fig8B_"
                },
            #"expertise_level": {
            #    "labels": ["Newcomer", "Experienced"],
            #    "title": "Assessment of Scientific Claims by Experience",
            #    "fig_name_prefix" : "fig??_"
            #},
        }

all_comparisons = []

for variable in all_categorical_variables.keys():
    # Group by historical lab status
    if variable == "Historical lab after 1998" or variable == "F and L":
        leading_author_claims_to_plot = leading_author_claims[leading_author_claims['year'] > 1995]
    else:
        leading_author_claims_to_plot = leading_author_claims
    if variable == "F and L":
        # take into account only the author that published at least once once after 1998
        author_to_keep = leading_author_claims[leading_author_claims['year'] > 1998]['leading_author_key'].unique()
        leading_author_claims_to_plot = leading_author_claims_to_plot[leading_author_claims_to_plot['leading_author_key'].isin(author_to_keep)]


    # This is wrong because it tooks the first value of variable for each author
    # am = wrangling.create_author_metric(claim_df = leading_author_claims_to_plot, 
    #                                     variable= "leading_author_key",
    #                                     other_col={"Name":('Name', 'first'),
    #                                                 variable:(variable, 'first')},)
    # if variable != "Historical lab after 1998":
    #     am = am[(am["Major claims"] >= 6) & (author_metrics["Articles"] >= 2)]
    # var_grouped = wrangling.group_categorical_variable(df=am, variable=variable,)
    
    # This was wrong because it overcounted claims when an author had different values for the same variable Utiliser pour les 2025-04-09
    # also this was filtered to only include authors with at least 6 major claims...
    #var_grouped2 = wrangling.prepare_categorical_variable_data(df=leading_author_claims, 
    #                                                author_metrics=author_metrics_to_plot, 
    #                                                variable=variable, 
    #                                                key_col='leading_author_key',
    #                                                assessment_columns=assessment_columns)


    var_grouped = wrangling.create_author_metric(claim_df = leading_author_claims_to_plot, 
                                        variable= variable,
                                        other_col={"Leading Author Name":('Leading Author Name', 'first'), "n_authors":('leading_author_key', 'nunique')}).set_index(variable)
    
    labels = all_categorical_variables[variable]["labels"]
    #if set(var_grouped.index) == set(labels):
    actual_groups = list(var_grouped.index)
    # Print to check if needed:
    print("Actual groups found:", actual_groups)
    if len(actual_groups) == 2:
        sentence, summary = stat_lib.report_categorical_comparison(var_grouped, 
                                                                        actual_groups, 
                                                                        outcome='Challenged',
                                                                        what_str=f"Leading Author {variable} ")
        print("\n"+sentence+"\n")
        all_comparisons.append({'Variable': variable, **summary})

        if variable == "Continuity":
            sentence, summary = stat_lib.report_categorical_comparison(var_grouped, 
                                                                        actual_groups, 
                                                                        outcome='Unchallenged',
                                                                        what_str=f"Leading Author {variable} ")
            print("FOR UNCHALLENGED:\n"+sentence+"\n")
            print("Summary for unchallenged claims:")
            print(summary)


    explain_df = leading_author_claims_to_plot.groupby(["Leading Author Name", variable]).agg(**{
        "Major claims":('id', 'count'),
        "Articles":('article_id', 'nunique')
        },
    ).reset_index().pivot(index="Leading Author Name", columns=variable, values="Major claims")
    explain_df.to_csv(f"figures/figure_tables/{all_categorical_variables[variable]['fig_name_prefix']}categorical_{variable}.csv", 
                    index=True, index_label=f"Name/{variable}", sep=";")

    # Calculate proportions
    for col in assessment_columns:
        var_grouped[f'{col}_prop'] = var_grouped[col] / var_grouped['Major claims']

    print(f"Summary of {variable}:")
    print(var_grouped[['Major claims', 'Articles', 
                        'Verified_prop', 'Challenged_prop', 'Unchallenged_prop', 'n_authors']])

    fig, ax = plot_info.create_horizontal_bar_chart(var_grouped, 
                                                    show_p_value=False, 
                                                    labels_map=labels, 
                                                    title=all_categorical_variables[variable]["title"],
                                                    other_n={"authors": "n_authors"})

    plt.savefig(f"figures/{all_categorical_variables[variable]['fig_name_prefix']}categorical_{variable}.png", dpi=300, bbox_inches='tight')

comparison_df = pd.DataFrame(all_comparisons)
comparison_df.to_csv("figures/figure_tables/summary_categorical_comparisons_leading.csv", sep=";", index=False)



# %%
to_plot = author_metrics.copy()

# %%
to_plot['Articles'].sort_values(ascending=False)

# %%

# Create article count bins
article_bins = [.5, 3.5, 7.5, 14.5,  float('inf')]
bin_labels =     ['<=3', '4-7', '7-14', '15+']
to_plot['Article_bin'] = pd.cut(to_plot['Articles'], bins=article_bins, labels=bin_labels)

# Create the boxplot
ax = sns.boxplot(x='Article_bin', y='Challenged prop', data=to_plot)
#ax = sns.stripplot(x='Article_bin', y='Challenged prop', data=to_plot, alpha=0.3, jitter=True, jitter_size=.95, color='black')

# Add individual data points
#sns.stripplot(x='Article_bin', y='Challenged prop', data=to_plot, 
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
#plt.savefig('figures/fig5C-nb_article_V1.png', dpi=300, bbox_inches='tight')

# %%
to_plot

# %%
author_metrics

# %% [markdown]
# ### Continuous Variables

# %%
# just take the more than 2 articles annd 6 claims
to_plot = author_metrics.copy()
min_articles = 2
min_claims = 6
to_plot = to_plot[to_plot['Articles'] >= min_articles]
to_plot = to_plot[to_plot['Major claims'] >= min_claims]
fig1, ax1 = plot_info.create_challenged_vs_unchallenged_scatter(to_plot, name_col="Leading Author Name")
print("This version of fig 9B is not the final one.")
plt.savefig('figures/fig9B.png', dpi=300, bbox_inches='tight')

# %%
to_plot

# %%

fig2, ax2 = plot_info.create_challenged_vs_articles_scatter(to_plot, name_col="Leading Author Name")
plt.savefig(f'figures/fig7B_scatterA.png', dpi=300, bbox_inches='tight')

# %%


# %%
to_plot["Major claims"].value_counts().sort_index().plot(kind='bar')    


# %%
to_plot["Articles"].value_counts().sort_index().plot(kind='bar')    

# %% [markdown]
# 

# %%
# Example usage for general scatter plot:
fig, ax = plot_info.create_publication_scatter(
    to_plot,
    x_var='Unchallenged prop', 
    y_var='Challenged prop',
    size_var='Articles', 
    title='Challenged vs. Unchallenged Claims by Author',
    x_label='Proportion of Unchallenged Claims',
    y_label='Proportion of Challenged Claims',
    annotate_top_n=5,
    name_col='Leading Author Name',
)
# plt.savefig(f'figures/fig6B_scatterB', dpi=300, bbox_inches='tight')

# %% [markdown]
# ### Things that uses both

# %%
first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")

# %%
all_claims = pd.merge(first_author_claims, leading_author_claims, on='id', how='outer', suffixes=('_first', '_last'))

# %%
import preprocess_utils


# %%


# %% [markdown]
# ## B. Binary variables

# %%
first_papers_year = {}
first_lh_or_fh_papers_year = {}
first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")

claims_sorted = preprocess_utils.build_author_key(leading_author_claims, "authors_txt", "all_authors_key")
claims_sorted = claims_sorted.sort_values(by=['year']).reset_index(drop=True)
claims_sorted["year"].plot()
for lh in leading_author_claims["leading_author_key"].unique():
    found = False
    for idx, row in claims_sorted.iterrows():
        all_aut = row["all_authors_key"]
        authors_list = [aut.strip() for aut in all_aut.split(";")]
        if lh in authors_list:
            first_papers_year[lh] = row["year"]
            found = True
            break  # Break out of the inner loop once we find the first paper
    if not found:
        print(f"Warning: No papers found for author {lh}")
    
    # Use skipna=True to ignore NaN/NA values, and check for empty series
    fh_series = first_author_claims[first_author_claims["first_author_key"] == lh]["year"]
    lh_series = claims_sorted[claims_sorted["leading_author_key"] == lh]["year"]
    
    fh_year = fh_series.min(skipna=True) if not fh_series.empty else None
    lh_year = lh_series.min(skipna=True) if not lh_series.empty else None
    
    # Check if the results are valid (not NaN/NA)
    fh_year = fh_year if pd.notna(fh_year) else None
    lh_year = lh_year if pd.notna(lh_year) else None
    
    if fh_year is not None and lh_year is not None:
        first_lh_or_fh_papers_year[lh] = min(fh_year, lh_year)
    elif fh_year is not None:
        first_lh_or_fh_papers_year[lh] = fh_year
    elif lh_year is not None:
        first_lh_or_fh_papers_year[lh] = lh_year
    else:
        print(f"Warning: No paper found for author {lh} in either first or leading author roles")                
first_papers_year = pd.DataFrame.from_dict(first_papers_year, orient='index', columns=['first_paper_year'])
first_lh_or_fh_papers_year = pd.DataFrame.from_dict(first_lh_or_fh_papers_year, orient='index', columns=['first_lh_or_fh_paper_year'])
first_papers_year = first_papers_year.merge(first_lh_or_fh_papers_year,  left_index=True, right_index=True, how='outer')

first_papers_year.plot()
# save it for the statistical analysis
first_papers_year.to_csv("preprocessed_data/lh_first_papers_year.csv", sep=";", index_label="leading_author_key")
# %%
to_plot = pd.merge(first_papers_year, author_metrics, left_index=True, right_on='leading_author_key', how='right')

# %%
fig, ax = plot_info.create_publication_scatter(
    to_plot,
    x_var='first_lh_or_fh_paper_year', 
    y_var='Challenged prop',
    size_var='Articles', 
    title='proportion Challenged vs. year of first paper',
    x_label="year of entry in the field",
    y_label="Proportion of challenged claims",
    annotate_top_n=50,
    name_col='Leading Author Name',
)
ax.set_ylim(0, .7)
ax.vlines(x=1995, ymin=0, ymax=.7, color='grey', linestyle='--')
plt.savefig(f'figures/fig8A_YearVSRepro', dpi=300, bbox_inches='tight')

# %% [markdown]
# ## Paper III, analyse interview

# %%
all_claims[all_claims["first_author_key"] == "stoven s"]

# %%
all_claims[all_claims["leading_author_key"] == "stoven s"]

# %%
inter_accepted = pd.DataFrame(
    [ "Stöven S", "Meister M", "Royet J", "Ligoxygakis P", "Lazzaro BP", "Bulet P", "Brennan CA", "Ferrandon D", "Engström Y", "Silverman N", "Markus R", 
    "Nicolas E", "Imler JL", "Govind S", "Dionne MS", "Watnick PI", "Apidianakis Y",],
    columns=["Name"])
inter_declined =  pd.DataFrame(["Foley E", "Lee WJ", "Schneider DS", "Kimbrell DA", "Kurata S", "Ip YT", "Fauvarque MO", "Wu LP"],
    columns=["Name"])
inter_excused =  pd.DataFrame(["Wasserman SA", "Kanuka H"],
    columns=["Name"])

all_inter = pd.concat([inter_accepted, inter_declined, inter_excused]).reset_index(drop=True)
all_inter = preprocess_utils.build_author_key(all_inter, "Name", "author_key")
all_inter = preprocess_utils.clean_author_keys(all_inter, "author_key")
all_inter['Major claims'] = 0
all_inter[assessment_columns] = 0

all_claims = preprocess_utils.build_author_key(all_claims, "authors_txt_first", "all_authors_key")
all_claims = preprocess_utils.clean_author_keys(all_claims, "all_authors_key")


for i, row in all_inter.iterrows():
    aut = row["author_key"]
    found = False
    for idx, row2 in all_claims.iterrows():
        all_aut = row2["all_authors_key"]
        authors_list = [aut2.strip() for aut2 in all_aut.split(";")]
        #if aut in authors_list:
        if (aut in row2["leading_author_key"]) or (aut in str(row2["first_author_key"])):
                all_inter.loc[i,'Major claims'] += 1
                this_ass = row2["assessment_type_grouped_last"]
                all_inter.loc[i, this_ass] += 1
                found = True
    if not found:
        print(f"Warning: No papers found for author {aut}")

for col in assessment_columns:
    all_inter[f'{col}_prop'] = all_inter[col] / all_inter['Major claims']

all_inter["Interview"] = all_inter["Name"].apply(
    lambda x: "Accepted" if x in inter_accepted["Name"].values
    else "Declined" if x in inter_declined["Name"].values
    else "Excused" if x in inter_excused["Name"].values
    else None
)

# %%
all_inter

# %%
all_inter.to_csv("interview_assessment.csv", sep=";", index=False)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Filter and compute
plot_df = all_inter[all_inter["Interview"].notnull()].copy()
plot_df["% Challenged"] = 100 * plot_df["Challenged"] / plot_df["Major claims"]

# Plot
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid", font_scale=1.2)

# Swarmplot
sns.swarmplot(data=plot_df, x="Interview", y="% Challenged", size=7, color=".1", marker='o')

# Pointplot with updated API
sns.pointplot(
    data=plot_df,
    x="Interview",
    y="% Challenged",
    errorbar='sd',
    linestyle='none',
    capsize=0.2,
    markers="D",
    color="darkred",
    err_kws={'linewidth': 1.5}
)

# Aesthetic improvements
plt.ylim(0, plot_df["% Challenged"].max() + 10)  # ensure y-axis is positive and covers full range
plt.title("% of Challenged Claims by Interview Category", fontsize=16)
plt.xlabel("Interview Category")
plt.ylabel("% Challenged Claims")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Figure 6 – Gini (A) and Distribution (B) for leading authors

# %%
import matplotlib.gridspec as gridspec
from plot_info import MEDIUM_SIZE

# Filter identical to standalone plots (≥ 2 articles & ≥ 6 major claims)
to_plot_lead = author_metrics.copy()
#to_plot_lead = to_plot_lead[(to_plot_lead["Articles"] >= 2) &
#                            (to_plot_lead["Major claims"] >= 6)]

fig6 = plt.figure(figsize=(18, 6))
gs6  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)

ax6A = fig6.add_subplot(gs6[0])   # A – Lorenz / Gini
ax6B = fig6.add_subplot(gs6[1])   # B – distribution scatter

# Panel A: Lorenz curve + Gini
plot_info.plot_lorenz_curve(
    to_plot_lead,
    prop_column="Challenged prop",
    weight_column="Major claims",
    #print_top_txt=False,
    title="",
    ax=ax6A,
)
ax6A.set_title("A", loc="left", fontweight="bold", fontsize=28)

# Panel B: distribution scatter
plot_info.plot_author_irreproducibility_focused(
    to_plot_lead,
    title="",
    color_by="Verified prop",
    cmap="RdYlGn",
    most_challenged_on_right=True,
    name_col="Leading Author Name",
    annotate_top_n=0,
    ax=ax6B,
)
ax6B.set_title("B", loc="left", fontweight="bold", fontsize=28)

# Unify legend (take from panel B)
lg6 = ax6B.get_legend()
handles6, labels6 = lg6.legend_handles, [t.get_text() for t in lg6.get_texts()]
lg6.remove()

ax6A.legend(
    handles6,
    labels6,
    loc="upper right",
    frameon=True,
    ncol=1,
    fontsize=MEDIUM_SIZE,
)

fig6.tight_layout()
fig6.savefig("figures/fig6_AB_leading_author_horizontal.png",
             dpi=300,
             bbox_inches="tight")
print("Saved → figures/fig6_AB_leading_author_horizontal.png")

 # %% [markdown]
# ### Figure 7 – Leading-author characteristics (A–C)

# %%
import matplotlib.gridspec as gridspec
from plot_info import MEDIUM_SIZE

# ── choose the two categorical variables for A & B ─────────────────────
varA = "Leading Author Sex"
varB = "Junior Senior"

label_mapA = None                                  # default labels
label_mapB = all_categorical_variables[varB]["labels"]

# ----------------------------------------------------------------------
# Prepare grouped data for each variable (identical to earlier loops)
def _make_group(var, lbl_map):
    g = wrangling.create_author_metric(
        claim_df=leading_author_claims,
        variable=var,
        other_col={"n_authors": ('leading_author_key', 'nunique')}
    ).set_index(var)
    for col in plot_info.assessment_columns:
        g[f"{col}_prop"] = g[col] / g["Major claims"]
    return g, (lbl_map or {v: v for v in g.index})

grpA, label_mapA = _make_group(varA, label_mapA)
grpB, label_mapB = _make_group(varB, label_mapB)

# ----------------------------------------------------------------------
# Layout: 2 rows × 2 cols – left col 40 %, right col 60 %
fig7 = plt.figure(figsize=(15, 10))
gs7  = gridspec.GridSpec(
    2, 2,
    width_ratios=[0.4, 0.6],
    height_ratios=[0.5, 0.5],
    wspace=0.35,
    hspace=0.25
)

ax7A = fig7.add_subplot(gs7[0, 0])                # top-left
ax7B = fig7.add_subplot(gs7[1, 0], sharex=ax7A)   # bottom-left (share y)
ax7C = fig7.add_subplot(gs7[:, 1])                # right (span rows)

# ── Panel A – Sex vertical bar ─────────────────────────────────────────
plot_info.create_horizontal_bar_chart(
    grpA,
    title="",
    labels_map=label_mapA,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    pct_axis_label="% of Claims",
    group_axis_label=varA,
    ax=ax7A,
)
ax7A.set_title("A", loc="left", fontweight="bold", fontsize=24)

# Capture legend handles from A
lg7 = ax7A.get_legend()
handles7 = lg7.legend_handles
labels7  = [t.get_text() for t in lg7.get_texts()]
lg7.remove()

# ── Panel B – Junior/Senior vertical bar ──────────────────────────────
plot_info.create_horizontal_bar_chart(
    grpB,
    title="",
    labels_map=label_mapB,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    pct_axis_label="% of Claims",
    group_axis_label=varB,
    ax=ax7B,
)
ax7B.set_title("B", loc="left", fontweight="bold", fontsize=24)
# Remove legend from B (if any)
if ax7B.get_legend():
    ax7B.get_legend().remove()

# ── Panel C – scatter (full right) ────────────────────────────────────
scatter_df = author_metrics.copy()
#scatter_df = scatter_df[(scatter_df["Articles"] >= 2) & (scatter_df["Major claims"] >= 6)]

plot_info.create_challenged_vs_articles_scatter(
    scatter_df,
    annotate_top_n=8,
    title="",
    size_mult=100,
    #name_col="Leading Author Name",
    ax=ax7C,
)
ax7C.set_title("C", loc="left", fontweight="bold", fontsize=24)

# ── Unified legend in upper-right of panel A ──────────────────────────
ax7A.legend(
    handles7,
    labels7,
    loc="upper right",
    frameon=True,
    ncol=1,
    fontsize=MEDIUM_SIZE,
)

fig7.tight_layout()
fig7.savefig("figures/fig7_ABC_leading_author_layout.png",
             dpi=300, bbox_inches="tight")
print("Saved → figures/fig7_ABC_leading_author_layout.png")

# %% [markdown]
# ### Figure 8 – Patterns of irreproducibility by last author according to time of starting their lab

# %%
import matplotlib.gridspec as gridspec
from plot_info import MEDIUM_SIZE

# Prepare data for panel A - year vs reproducibility (already exists in the code above)
# Apply the same filtering as in the individual panels (≥2 articles & ≥6 major claims)
to_plot_year = pd.merge(first_papers_year, author_metrics, left_index=True, right_on='leading_author_key', how='right')
#to_plot_year = to_plot_year[(to_plot_year["Articles"] >= 2) & (to_plot_year["Major claims"] >= 6)]

# ── Choose the two categorical variables for B & C using the same approach as Figure 7 ──
varB = "F and L"
varC = "Historical lab after 1998"

# Use the same _make_group function as Figure 7, but with proper data filtering
def _make_group_fig8(var, lbl_map, claims_data):
    g = wrangling.create_author_metric(
        claim_df=claims_data,
        variable=var,
        other_col={"n_authors": ('leading_author_key', 'nunique')}
    ).set_index(var)
    for col in plot_info.assessment_columns:
        g[f"{col}_prop"] = g[col] / g["Major claims"]
    return g, (lbl_map or {v: v for v in g.index})

# Prepare filtered data for each variable as done in the original categorical loop
# For "F and L" - filter after 1995 and only authors who published after 1998
leading_author_claims_B = leading_author_claims[leading_author_claims['year'] > 1995]
author_to_keep_B = leading_author_claims[leading_author_claims['year'] > 1998]['leading_author_key'].unique()
leading_author_claims_B = leading_author_claims_B[leading_author_claims_B['leading_author_key'].isin(author_to_keep_B)]

# For "Historical lab after 1998" - filter after 1995
leading_author_claims_C = leading_author_claims[leading_author_claims['year'] > 1995]

# Generate grouped data using the same method as Figure 7
grpB, label_mapB = _make_group_fig8(varB, all_categorical_variables[varB]["labels"], leading_author_claims_B)
grpC, label_mapC = _make_group_fig8(varC, all_categorical_variables[varC]["labels"], leading_author_claims_C)

# Layout: 2 rows × 2 cols – left col 40%, right col 60%
# Reduce height ratios to make space for legend at bottom
fig8 = plt.figure(figsize=(15, 10))
gs8 = gridspec.GridSpec(
    3, 2,
    width_ratios=[0.4, 0.6],
    height_ratios=[0.4, 0.4,0.2],  # Reduced from 0.5 to make space for legend
    wspace=0.35,
    hspace=0.2  # Reduced spacing
)

ax8A = fig8.add_subplot(gs8[0, 0])                # top-left
ax8B = fig8.add_subplot(gs8[1, 0], sharex=ax8A)   # bottom-left
ax8C = fig8.add_subplot(gs8[:, 1])                # right (span rows)

# Panel A - Year vs reproducibility scatter plot
# Create scatter plot without size legend first
scatter = ax8C.scatter(
    to_plot_year['first_lh_or_fh_paper_year'],
    to_plot_year['Challenged prop'],
    s=to_plot_year['Articles'] * 15,  # Size by Articles
    c='#3498db',  # Default blue
    alpha=0.7,
    edgecolors='white',
    linewidth=0.5
)

# # Add regression line
# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(
#     to_plot_year['first_lh_or_fh_paper_year'], to_plot_year['Challenged prop']
# )
# x_line = np.linspace(to_plot_year['first_lh_or_fh_paper_year'].min(), to_plot_year['first_lh_or_fh_paper_year'].max(), 100)
# y_line = intercept + slope * x_line
# ax8C.plot(x_line, y_line, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2)
# 
# # Add regression statistics
# stats_text = f"$r^2$ = {r_value**2:.3f}\\n"
# stats_text += f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
# stats_text += "\\n" + f"y = {slope:.3f}x + {intercept:.3f}"
# ax8C.text(0.05, 0.95, stats_text, transform=ax8C.transAxes, 
#            va='top', ha='left', fontsize=12, 
#            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
# 
# Create size legend on the scatter plot itself (middle left)
sizes = [5, 10, 20, 30]
legend_elements = []
for size in sizes:
    legend_elements.append(plt.scatter([], [], s=size*15, color='gray', alpha=0.7, 
                                     edgecolors='white', linewidth=0.5, label=f"{size}"))

legend = ax8C.legend(
    legend_elements, [f"{size}" for size in sizes],
    title="Number of Articles",
    loc='center left',  # Middle left of the scatter plot
    frameon=True,
    framealpha=0.9,
    edgecolor='lightgray',
    handletextpad=2,
    labelspacing=1
)
legend.get_title().set_fontweight('bold')

ax8C.set_ylim(0, 0.7)
ax8C.vlines(x=1995, ymin=0, ymax=0.7, color='grey', linestyle='--')
ax8C.set_xlabel("Year of entry in the field", fontweight='bold')
ax8C.set_ylabel("Proportion of challenged claims", fontweight='bold')
ax8C.yaxis.set_major_formatter(plot_info.PercentFormatter(1.0))
ax8C.grid(linestyle='--', alpha=0.3)
ax8C.set_axisbelow(True)
ax8C.spines['top'].set_visible(False)
ax8C.spines['right'].set_visible(False)

# Remove the automatic title that the function creates
ax8C.set_title("", fontsize=1)  # Clear any title
ax8C.text(0.02, 0.98, "A", transform=ax8C.transAxes, fontweight="bold", fontsize=24, va="top")

# Panel B - F and L categorical analysis
plot_info.create_horizontal_bar_chart(
    grpB,
    title="",
    labels_map=label_mapB,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    pct_axis_label="% of Claims",
    #group_axis_label="Previous mentee experience",
    ax=ax8A,
)
ax8A.set_title("B", loc="left", fontweight="bold", fontsize=24)

# Capture legend handles from B
lg8 = ax8A.get_legend()
handles8 = lg8.legend_handles
labels8 = [t.get_text() for t in lg8.get_texts()]
lg8.remove()

# Panel C - Historical lab categorical analysis
plot_info.create_horizontal_bar_chart(
    grpC,
    title="",
    labels_map=label_mapC,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    pct_axis_label="% of Claims",
    #group_axis_label="Laboratory tradition",
    ax=ax8B,
)
ax8B.set_title("C", loc="left", fontweight="bold", fontsize=24)
# Remove legend from C (if any)
if ax8B.get_legend():
    ax8B.get_legend().remove()

# Place the common legend under panel C in 2 columns
fig8.legend(
    handles8,
    labels8,
    loc="lower center",
    bbox_to_anchor=(0.25, 0.1),  # Position at bottom with some space
    frameon=True,
    ncol=2,  # 2 columns as requested
    fontsize=MEDIUM_SIZE,
    #title="Assessment Category",
    title_fontsize=MEDIUM_SIZE
)

fig8.tight_layout()
fig8.savefig("figures/fig8_ABC_time_patterns_layout.png",
             dpi=300, bbox_inches="tight")
print("Saved → figures/fig8_ABC_time_patterns_layout.png")

# %% [markdown]
# ### Figure 9 – Continuity analysis with vertical bars and scatter

# %%
# Prepare data for Figure 9
varA_fig9 = "Continuity"
varA_fig9_data = leading_author_claims  # No special filtering for Continuity

# Generate grouped data for Continuity
grpA_fig9, label_mapA_fig9 = _make_group_fig8(varA_fig9, all_categorical_variables[varA_fig9]["labels"], varA_fig9_data)

# Use the same scatter data as other figures (filtered)
scatter_df_fig9 = pd.merge(first_papers_year, author_metrics, left_index=True, right_on='leading_author_key', how='right').copy()
scatter_df_fig9 = scatter_df_fig9[(scatter_df_fig9["first_lh_or_fh_paper_year"] >= 1995)]

# Layout: 2 rows × 2 cols – legend above stackplot, scatter takes full right side
fig9 = plt.figure(figsize=(15, 8))
gs9 = gridspec.GridSpec(
    2, 2,
    width_ratios=[0.25, 0.75],  # Left 25%, right 75%
    height_ratios=[0.3, 0.7],   # Top 30% for legend, bottom 70% for stackplot
    wspace=0.2,
    hspace=0.1
)

ax9_legend = fig9.add_subplot(gs9[0, 0])  # top-left - legend space
ax9A = fig9.add_subplot(gs9[1, 0])       # bottom-left - categorical (70% height)
ax9B = fig9.add_subplot(gs9[:, 1])       # right - scatter (spans both rows)

# Panel A - Continuity vertical bar chart
plot_info.create_horizontal_bar_chart(
    grpA_fig9,
    title="",
    labels_map=label_mapA_fig9,
    show_p_value=False,
    other_n={"authors": "n_authors"},
    pct_axis_label="% of Claims",
    group_axis_label="",
    orientation="vertical",  # Make it vertical
    ax=ax9A,
)
ax9A.set_title("A", loc="left", fontweight="bold", fontsize=24)

# Capture legend handles from A
lg9 = ax9A.get_legend()
handles9 = lg9.legend_handles
labels9 = [t.get_text() for t in lg9.get_texts()]
lg9.remove()

# Panel B - Challenged vs Unchallenged scatter
plot_info.create_challenged_vs_unchallenged_scatter(
    scatter_df_fig9,
    annotate_top_n=0,
    title="",
    size_mult=100,
    name_col="Leading Author Name",
    ax=ax9B,
)
ax9B.set_title("B", loc="left", fontweight="bold", fontsize=24)

# Place the legend vertically above the stacked plot
ax9_legend.axis('off')  # Hide the axes

# Position the legend in the top-left area, above the stacked plot
legend = ax9_legend.legend(
    handles9,
    labels9,
    loc="center",
    frameon=True,
    ncol=1,  # Single column (vertical)
    fontsize=MEDIUM_SIZE,
    #title="Assessment Category",
    title_fontsize=MEDIUM_SIZE,
    bbox_to_anchor=(0.5, 0.5)
)

fig9.tight_layout()
fig9.savefig("figures/fig9_AB_continuity_layout.png",
             dpi=300, bbox_inches="tight")
print("Saved → figures/fig9_AB_continuity_layout.png")

# %%
# Statistical analysis: Compare challenged claims before vs after 1995
print("\n=== STATISTICAL ANALYSIS: PRE-1995 vs POST-1995 ENTRY ===\n")

# Filter to_plot_year to remove rows with missing first_lh_or_fh_paper_year
analysis_data = to_plot_year.dropna(subset=['first_lh_or_fh_paper_year']).copy()

# Create binary variable for entry before/after 1995
analysis_data['entry_before_1995'] = analysis_data['first_lh_or_fh_paper_year'] < 1995

# Print summary statistics
print("Sample sizes:")
pre_1995_count = len(analysis_data[analysis_data['entry_before_1995'] == True])
post_1995_count = len(analysis_data[analysis_data['entry_before_1995'] == False])
print(f"Authors entering before 1995: {pre_1995_count}")
print(f"Authors entering 1995 or after: {post_1995_count}")

# Group by entry period and calculate challenged claims
entry_summary = analysis_data.groupby('entry_before_1995').agg({
    'Challenged': 'sum',           # Total challenged claims
    'Major claims': 'sum',         # Total claims
    'leading_author_key': 'count'  # Number of authors
}).rename(columns={'leading_author_key': 'Authors'})

# Add challenged proportion
entry_summary['Challenged_prop'] = entry_summary['Challenged'] / entry_summary['Major claims']

print(f"\nSummary by entry period:")
print(entry_summary)

# Report proportions with confidence intervals
print(f"\n1. PROPORTIONS BY ENTRY PERIOD:")
print("-" * 40)

for period, is_before in [(False, "1995 or after"), (True, "before 1995")]:
    period_data = analysis_data[analysis_data['entry_before_1995'] == period]
    challenged_count = period_data['Challenged'].sum()
    total_count = period_data['Major claims'].sum()
    
    proportion_report = stat_lib.report_proportion(
        successes=challenged_count,
        total=total_count,
        end_sentence=f"of claims from authors entering {is_before} were challenged."
    )
    print(f"Authors entering {is_before}: {proportion_report}")

# Statistical comparison using report_categorical_comparison
print(f"\n2. STATISTICAL COMPARISON:")
print("-" * 30)

# Prepare data for comparison (need to have the right index structure)
comparison_data = entry_summary.copy()
comparison_data.index = ['Post-1995', 'Pre-1995']  # Rename for clarity

comparison_sentence, comparison_summary = stat_lib.report_categorical_comparison(
    var_grouped=comparison_data,
    labels=['Pre-1995', 'Post-1995'],  # Pre-1995 vs Post-1995
    outcome='Challenged',
    what_str="Entry period (pre-1995 vs post-1995)"
)

print(comparison_sentence)

# Additional analysis: Show distribution by decade
print(f"\n3. ADDITIONAL ANALYSIS BY DECADE:")
print("-" * 35)

# Create decade bins
analysis_data['decade'] = pd.cut(analysis_data['first_lh_or_fh_paper_year'], 
                                bins=[1970, 1980, 1990, 1995, 2000, 2010], 
                                labels=['1970s', '1980s', '1990-1994', '1995-1999', '2000s'],
                                include_lowest=True)

decade_summary = analysis_data.groupby('decade').agg({
    'Challenged': 'sum',
    'Major claims': 'sum',
    'leading_author_key': 'count'
}).rename(columns={'leading_author_key': 'Authors'})

decade_summary['Challenged_prop'] = decade_summary['Challenged'] / decade_summary['Major claims']

print("Summary by decade of entry:")
for decade in decade_summary.index:
    if pd.notna(decade):
        challenged = decade_summary.loc[decade, 'Challenged']
        total = decade_summary.loc[decade, 'Major claims']
        authors = decade_summary.loc[decade, 'Authors']
        prop = decade_summary.loc[decade, 'Challenged_prop']
        print(f"{decade}: {authors} authors, {challenged}/{total} challenged ({prop:.1%})")

# %%
