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

claims = pd.read_csv('preprocessed_data/claims_db_truncated_for_llm.csv')
claims = claims[(claims['assertion_type'] == 'main_claim')]
claims = claims[['article_id', 'assessment_type']]

# Merge citations and claims dataframes
merged_df = pd.merge(claims, citations, 
              left_on='article_id', 
              right_on='Article ID', 
              how='right').drop(columns=['article_id', 'Article ID'])


# %%
merged_df

# %%


    
# Create simplified categories
category_mapping = {
    'Verified': 'Verified',
    'Verified by same authors': 'Verified',
    'Verified by reproducibility project': 'Verified',
    'Challenged': 'Challenged',
    'Challenged by reproducibility project': 'Challenged',
    'Challenged by same authors': 'Challenged',
    'Unchallenged': 'Unchallenged',
    'Unchallenged, logically consistent': 'Unchallenged',
    'Unchallenged, logically inconsistent': 'Unchallenged'
}

# Map categories and filter for main categories
merged_df['simplified_assessment'] = merged_df['assessment_type'].map(category_mapping)
merged_df = merged_df[merged_df['simplified_assessment'].notna()]


def calculate_citation_statistics(df):
    """
    Calculate citation statistics for each category
    """
    citation_cols = [col for col in df.columns if 'Citations' in col]
    stats = {}
    
    for category in df['simplified_assessment'].unique():
        category_data = df[df['simplified_assessment'] == category]
        
        # Calculate mean citations per year
        yearly_means = category_data[citation_cols].mean()
        
        # Calculate standard error
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
        index='simplified_assessment',
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

# Create visualizations
plot_citation_patterns(stats)
plt.savefig('figures/fig12_citation_patterns.png')

    
#create_citation_heatmap(merged_df)
#plt.savefig('figures/citation_heatmap.png')

    
# Calculate and print summary statistics
summary = pd.DataFrame({
    category: {
        'Total Citations': merged_df[merged_df['simplified_assessment'] == category][
            [col for col in merged_df.columns if 'Citations' in col]
        ].sum().sum(),
        'Average Citations per Year': merged_df[merged_df['simplified_assessment'] == category][
            [col for col in merged_df.columns if 'Citations' in col]
        ].mean().mean(),
        'Number of Claims': len(merged_df[merged_df['simplified_assessment'] == category])
    }
    for category in merged_df['simplified_assessment'].unique()
}).T

print("\nSummary Statistics:")
print(summary)







# %%


# %%



