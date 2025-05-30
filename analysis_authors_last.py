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
first_papers_year = pd.DataFrame.from_dict(first_papers_year, orient='index', columns=['first_paper_year'])

# %%
to_plot = pd.merge(first_papers_year, author_metrics, left_index=True, right_on='leading_author_key', how='right')

# %%
fig, ax = plot_info.create_publication_scatter(
    to_plot,
    x_var='first_paper_year', 
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


