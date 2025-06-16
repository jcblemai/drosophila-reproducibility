# %%
# read an xlx file in pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the data
citations = pd.read_excel('input_data/2025-02-14_last_xlsx/citation_counts.xlsm')

citations.to_csv("input_data/2025-02-14_last_xlsx/citation_counts.csv", index=False)
citations = pd.read_csv('input_data/2025-04-24/citation_counts_15year.txt', sep='\t')


citations = citations[['Article ID', 'Citations (year+0)', 'Citations (year+1)',
    'Citations (year+2)', 'Citations (year+3)', 'Citations (year+4)',
    'Citations (year+5)', 'Citations (year+6)', 'Citations (year+7)',
    'Citations (year+8)', 'Citations (year+9)', 'Citations (year+10)',
    'Citations (year+11)', 'Citations (year+12)', 'Citations (year+13)',
    'Citations (year+14)', 'Citations (year+15)']]
    
citations.to_csv("preprocessed_data/citation_counts_short.csv", index=False)

first_author_claims = pd.read_csv("preprocessed_data/first_author_claims.csv")


leading_author_claims = pd.read_csv("preprocessed_data/leading_author_claims.csv")
lh_first_papers_year = pd.read_csv("preprocessed_data/lh_first_papers_year.csv", sep=";")
leading_author_claims = pd.merge(leading_author_claims, lh_first_papers_year, how="left", on="leading_author_key")

first_author_claims = first_author_claims[[
                                            'id', 
                                            'first_author_key',
                                            'First Author Sex',
                                            'PhD Post-doc',
                                            'First Author Become a PI'
                                            ]]

leading_author_claims = leading_author_claims[[
                                            'id',  
                                            'leading_author_key', 
                                            'Historical lab after 1998', 
                                            'Continuity',
                                            'Leading Author Sex',
                                            'Junior Senior',
                                            "F and L",
                                            'first_lh_or_fh_paper_year', 
                                            #  ~~~ Paper covariates
                                            'year', 
                                            'year_binned',
                                            'journal_category',
                                            'ranking_category', 
                                            'article_id',
                                            # ~~~ outcome variables
                                            'assessment_type_grouped',
                                            ]]

leading_author_claims["first_paper_before_1995"] = leading_author_claims["first_lh_or_fh_paper_year"] < 1995


all_covar = pd.merge(first_author_claims, leading_author_claims, how="left", left_on="id", right_on="id", suffixes=("", "_lh"))
all_covar = all_covar.drop(all_covar.filter(regex='_lh$').columns, axis=1)

claims = all_covar[['article_id', 'year', 'journal_category','assessment_type_grouped']].copy()

# Merge citations and claims dataframes
merged_df = pd.merge(claims, citations, 
              left_on='article_id', 
              right_on='Article ID', 
              how='inner')


# Group by article and determine overall assessment per article
def determine_article_assessment(assessments):
    """
    Determine article assessment based on hierarchy:
    1. If any claim is challenged -> Challenged
    2. Else if any claim is verified -> Verified  
    3. Else if any claim is unchallenged -> Unchallenged
    4. Else -> Unknown (filtered out)
    """
    assessments = assessments.dropna()
    if len(assessments) == 0:
        return None
    if 'Challenged' in assessments.values:
        return 'Challenged'
    elif 'Verified' in assessments.values:
        return 'Verified'
    elif 'Unchallenged' in assessments.values:
        return 'Unchallenged'
    else:
        return None

# Group by Article ID and get one row per article
article_assessments = merged_df.groupby('Article ID').agg({
    'assessment_type_grouped': determine_article_assessment,
    'year': 'first',  # Take first occurrence (should be same for all claims from same article)
    'journal_category': 'first',  # Take first occurrence (should be same for all claims from same article)
}).reset_index()

# Merge back with citation data (keep all citation columns)
citation_cols = [col for col in citations.columns if 'Citations' in col]
citations_for_merge = citations[['Article ID'] + citation_cols]

merged_df = pd.merge(article_assessments, citations_for_merge, on='Article ID', how='inner')

# Filter out unknown assessments
merged_df = merged_df[merged_df['assessment_type_grouped'].notna()]

# Rename Article ID to article_id for easier formula parsing
merged_df = merged_df.rename(columns={'Article ID': 'article_id'})


def calculate_citation_statistics(df, normalize=False):
    """
    Calculate citation statistics for each category
    """
    citation_cols = [f'Citations (year+{i})' for i in range(16)]
    stats = {}
    
    # Calculate total citations per calendar year if normalizing
    if normalize:
        total_by_year = {}
        for _, row in df.iterrows():
            if pd.notna(row['year']):
                pub_year = int(row['year'])
                for i in range(16):
                    calendar_year = pub_year + i
                    citations = row[citation_cols[i]]
                    if pd.notna(citations):
                        total_by_year[calendar_year] = total_by_year.get(calendar_year, 0) + citations
    
    for category in df['assessment_type_grouped'].unique():
        category_data = df[df['assessment_type_grouped'] == category].copy()
        
        if normalize:
            # Normalize by total citations in each calendar year
            for idx, row in category_data.iterrows():
                if pd.notna(row['year']):
                    pub_year = int(row['year'])
                    for i in range(16):
                        calendar_year = pub_year + i
                        citations = row[citation_cols[i]]
                        if pd.notna(citations) and calendar_year in total_by_year and total_by_year[calendar_year] > 0:
                            category_data.loc[idx, citation_cols[i]] = float(citations) / total_by_year[calendar_year]
        
        yearly_means = category_data[citation_cols].mean()
        yearly_se = category_data[citation_cols].sem()
        
        stats[category] = {
            'means': yearly_means,
            'se': yearly_se
        }
    
    return stats

def plot_citation_patterns(stats):
    """
    Create visualization of citation patterns
    """
    plt.figure(figsize=(12, 6))
    
    colors = {'Verified': '#2ecc71', 'Challenged': '#e74c3c', 'Unchallenged': '#3498db'}
    years = range(16)  # 0 to 10 years
    
    for category, data in stats.items():
        means = data['means']
        se = data['se']
        
        plt.plot(years, means, label=category, color=colors[category], linewidth=2)
        plt.fill_between(years, 
                        means - se, 
                        means + se, 
                        alpha=0.2, 
                        color=colors[category])
    
    plt.xlabel('Years since publication')
    plt.ylabel('Average number of citations')
    #plt.title('Citation Patterns by Assessment Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def create_citation_heatmap(df):
    """
    Create a heatmap showing citation patterns
    """
    citation_cols = [col for col in df.columns if 'Citations' in col]
    pivot_data = df.pivot_table(
        values=citation_cols,
        index='assessment_type_grouped',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 4))
    sns.heatmap(pivot_data, 
                cmap='YlOrRd', 
                annot=True, 
                fmt='.1f', 
                cbar_kws={'label': 'Average Citations'})
    plt.title('Citation Heatmap by Assessment Type')
    plt.xlabel('Years since publication')
    plt.ylabel('Assessment Type')
    
    return plt


    
    # Calculate statistics
stats = calculate_citation_statistics(merged_df)

# Calculate normalized statistics
stats_normalized = calculate_citation_statistics(merged_df, normalize=True)

# Create visualizations
plot_citation_patterns(stats)
plt.savefig('figures/fig12_citation_patterns.png')

plt.figure()
plot_citation_patterns(stats_normalized)
plt.ylabel('Normalized citations (proportion of total)')
plt.savefig('figures/fig12_citation_patterns_normalized.png')

    
#create_citation_heatmap(merged_df)
#plt.savefig('figures/citation_heatmap.png')

    
# Calculate and print summary statistics
summary = pd.DataFrame({
    category: {
        'Total Citations': merged_df[merged_df['assessment_type_grouped'] == category][
            [col for col in merged_df.columns if 'Citations' in col]
        ].sum().sum(),
        'Average Citations per Year': merged_df[merged_df['assessment_type_grouped'] == category][
            [col for col in merged_df.columns if 'Citations' in col]
        ].mean().mean(),
        'Number of Claims': len(merged_df[merged_df['assessment_type_grouped'] == category])
    }
    for category in merged_df['assessment_type_grouped'].unique()
}).T

print("\nSummary Statistics:")
print(summary)







# %%
year_cols       = [col for col in citations.columns if 'Citations (year+' in col]
years           = [int(col.split('+')[1].rstrip(')')) for col in year_cols]


# %%

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
# Configure matplotlib for better notebook display
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.ion()  # Turn on interactive mode for better notebook display
import matplotlib
matplotlib.use('inline')  # Use inline backend for Jupyter
import stat_lib
import pytensor, os
pytensor.config.cxx = "/usr/bin/clang++"
os.environ["CXXFLAGS"] = "-std=c++17"
import bambi as bmb
import patsy
import arviz as az


# ───────────────────────────────────────────────────────────────
# 0.  Preparation – cumulative 0–15-year citations per article
# ───────────────────────────────────────────────────────────────
cit_cols_0_15   = [f'Citations (year+{i})' for i in range(16)]
merged_df['cits_15y'] = merged_df[cit_cols_0_15].sum(axis=1)

# ───────────────────────────────────────────────────────────────
# 1.  Analysis dataframe (one row per article)
# ───────────────────────────────────────────────────────────────
mod_df = (
    merged_df[['cits_15y', 'assessment_type_grouped', 'journal_category', 'year']]
    .dropna()
    .assign(cits_15y = lambda d: d.cits_15y.astype(int))
)

# 3-knot spline for publication year
spl     = patsy.bs(mod_df['year'], df=3, include_intercept=False)
mod_df[['year_s1', 'year_s2', 'year_s3']] = pd.DataFrame(spl, index=mod_df.index)

# ───────────────────────────────────────────────────────────────
# 2.  Bambi model:  citations ~ assessment × journal + spl(year)
# ───────────────────────────────────────────────────────────────
formula = (
    "cits_15y ~ "
    "C(assessment_type_grouped, Treatment('Verified')) * "
    "C(journal_category,        Treatment('Low Impact'))   + "
    "year_s1 + year_s2 + year_s3 "
)

model = bmb.Model(
    formula,
    mod_df,
    family="negativebinomial",
    dropna=True
)

# intercept prior centred on log mean citation rate
mu_rate = mod_df.cits_15y.mean()
model.set_priors({'Intercept': bmb.Prior("Normal",
                                         mu=np.log(mu_rate+1e-3), sigma=1.5)})

idata = model.fit(draws=2000, tune=1000, chains=4, cores=4,
                  target_accept=0.9, random_seed=123)

# ───────────────────────────────────────────────────────────────
# 3.  Incidence-rate ratios (IRR) — print to screen
# ───────────────────────────────────────────────────────────────
# Get fixed effect variables (exclude intercept and random effects)
fixed_vars = [var for var in idata.posterior.data_vars if not ('_sigma' in var or '_offset' in var or 'alpha' in var)]
summ = az.summary(idata, var_names=fixed_vars, kind='stats')
summ['IRR']      = np.exp(summ['mean'])
summ['IRR_low']  = np.exp(summ['hdi_3%'])
summ['IRR_high'] = np.exp(summ['hdi_97%'])

print("\nIRR for total 0–15-y citations:")
print(summ[['IRR','IRR_low','IRR_high']].round(2))

# ───────────────────────────────────────────────────────────────
# 4.  Create formatted table and forest plot like statistical_analysis
# ───────────────────────────────────────────────────────────────

# Filter for main effects only (excluding intercept, splines, and interactions)
assessment_effects = summ[summ.index.str.contains('assessment_type_grouped') & 
                         ~summ.index.str.contains(':')]  # Exclude interactions
journal_effects = summ[summ.index.str.contains('journal_category') & 
                      ~summ.index.str.contains(':')]     # Exclude interactions

# Create formatted table using stat_lib
print("\n=== CITATION ANALYSIS RESULTS ===")
print("\n--- Assessment Type Effects ---")
assessment_formatted = stat_lib.format_results_table(assessment_effects, clean_variable_names=True)
print(assessment_formatted[['IRR', 'IRR_low', 'IRR_high']])

print("\n--- Journal Category Effects ---")
journal_formatted = stat_lib.format_results_table(journal_effects, clean_variable_names=True)
print(journal_formatted[['IRR', 'IRR_low', 'IRR_high']])

# Combine effects for forest plot
all_citation_effects = pd.concat([
    assessment_effects.assign(category='Assessment'),
    journal_effects.assign(category='Journal')
])

# Rename IRR columns to OR columns for stat_lib compatibility
plot_effects = all_citation_effects.copy()
plot_effects['OR'] = plot_effects['IRR']
plot_effects['OR_low'] = plot_effects['IRR_low'] 
plot_effects['OR_high'] = plot_effects['IRR_high']

# Create forest plot using stat_lib functions
fig, ax = stat_lib.create_forest_plot(plot_effects)
plt.savefig('figM3_citation_forest_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create elegant forest plot using forestplot package
ax_elegant = stat_lib.create_elegant_forest_plot(plot_effects, "Citation Analysis")
plt.savefig('figM3_citation_elegant_forest_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Save citation analysis tables
assessment_formatted.to_csv('tableM3_citation_assessment_effects.csv')
journal_formatted.to_csv('tableM3_citation_journal_effects.csv')

# Save combined effects table
all_citation_formatted = stat_lib.format_results_table(all_citation_effects, clean_variable_names=True)
all_citation_formatted.to_csv('tableM3_citation_all_effects.csv')
print("Saved citation analysis results to tableM3_*.csv files")

