# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('preprocessed_data/claims.csv')

# Filter for major claims
major_claims = df[df['assertion_type'] == 'major_claim']

# Create journal type categories
def categorize_journal(impact_factor):
    if pd.isna(impact_factor):
        return None
    if impact_factor >= 30:  # Nature, Science, Cell typically have very high IF
        return 'Trophy Journals'
    elif impact_factor >= 10:
        return 'High Impact'
    else:
        return 'Low Impact'

# Group assessment types
def group_assessment(assessment):
    if pd.isna(assessment) or assessment == 'Not assessed':
        return None
    if 'Verified' in assessment:
        return 'Verified'
    elif 'Challenged' in assessment:
        return 'Challenged'
    elif 'Mixed' in assessment:
        return 'Mixed'
    elif 'Partially verified' in assessment:
        return 'Partially Verified'
    elif 'Unchallenged' in assessment:
        return 'Unchallenged'
    else:
        return None

major_claims['journal_category'] = major_claims['impact_factor'].apply(categorize_journal)
major_claims['assessment_group'] = major_claims['assessment_type'].apply(group_assessment)

# Create pivot table
pivot_data = pd.pivot_table(
    major_claims[major_claims['journal_category'].notna() & major_claims['assessment_group'].notna()],
    values='num',
    index='journal_category',
    columns='assessment_group',
    aggfunc='count',
    fill_value=0
)

# Reorder index
pivot_data = pivot_data.reindex(['Low Impact', 'High Impact', 'Trophy Journals'])

# Calculate percentages
pivot_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Define colors for each assessment type
colors = {
    'Unchallenged': '#3498db',
    'Mixed': '#95a5a6',
    'Verified': '#2ecc71',
    'Partially Verified': '#f1c40f',
    'Challenged': '#e74c3c'
}

# Plot 1: Absolute numbers
bottom1 = np.zeros(len(pivot_data))
for col in ['Unchallenged', 'Mixed', 'Verified', 'Partially Verified', 'Challenged']:
    if col in pivot_data.columns:
        ax1.bar(pivot_data.index, pivot_data[col], bottom=bottom1, label=col, color=colors[col])
        # Add value labels
        for i in range(len(pivot_data.index)):
            if pivot_data[col][i] > 0:  # Only show non-zero values
                ax1.text(i, bottom1[i] + pivot_data[col][i]/2,
                        f'{int(pivot_data[col][i])}',
                        ha='center', va='center')
        bottom1 += pivot_data[col]

# Plot 2: Percentages
bottom2 = np.zeros(len(pivot_pct))
for col in ['Unchallenged', 'Mixed', 'Verified', 'Partially Verified', 'Challenged']:
    if col in pivot_pct.columns:
        ax2.bar(pivot_pct.index, pivot_pct[col], bottom=bottom2, label=col, color=colors[col])
        # Add percentage labels
        for i in range(len(pivot_pct.index)):
            if pivot_pct[col][i] > 5:  # Only show labels for segments > 5%
                ax2.text(i, bottom2[i] + pivot_pct[col][i]/2,
                        f'{pivot_pct[col][i]:.1f}%',
                        ha='center', va='center')
        bottom2 += pivot_pct[col]

# Customize the plots
ax1.set_title('Absolute Number of Major Claims\nby Journal Category and Assessment Type', pad=20)
ax1.set_xlabel('Journal Category')
ax1.set_ylabel('Number of Claims')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2.set_title('Distribution of Major Claims (%)\nby Journal Category and Assessment Type', pad=20)
ax2.set_xlabel('Journal Category')
ax2.set_ylabel('Percentage of Claims')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, title='Assessment Type',
          bbox_to_anchor=(1.02, 0.5), loc='center left')

# Adjust layout
plt.subplots_adjust(right=0.85)

# Print the raw numbers for reference
print("\nRaw numbers of claims by category:")
print(pivot_data)
print("\nPercentages by category:")
print(pivot_pct.round(1))

# save as png and pdf in figures/
plt.savefig('figures/claims_by_journal_and_assessment.png', bbox_inches='tight')
plt.savefig('figures/claims_by_journal_and_assessment.pdf', bbox_inches='tight')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('preprocessed_data/claims.csv')

# Filter for major claims
major_claims = df[df['assertion_type'] == 'major_claim'].copy()

# Group assessment types
def group_assessment(assessment):
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

# Create year bins
def bin_years(year):
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

major_claims['assessment_group'] = major_claims['assessment_type'].apply(group_assessment)
major_claims['year_group'] = major_claims['year'].apply(bin_years)

# Create pivot table
pivot_data = pd.pivot_table(
    major_claims[major_claims['year_group'].notna() & major_claims['assessment_group'].notna()],
    values='num',
    index='year_group',
    columns='assessment_group',
    aggfunc='count',
    fill_value=0
)

# Define year order
year_order = ['≤1991', '1992-1996', '1997-2001', '2002-2006', '2007-2011']
pivot_data = pivot_data.reindex(index=year_order)

# Calculate percentages
pivot_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Define colors and order
colors = {
    'Unchallenged': '#3498db',
    'Mixed': '#95a5a6',
    'Verified': '#2ecc71',
    'Partially Verified': '#f1c40f',
    'Challenged': '#e74c3c'
}

assessment_order = ['Unchallenged', 'Mixed', 'Verified', 'Partially Verified', 'Challenged']

# Plot 1: Absolute numbers
bottom1 = np.zeros(len(pivot_data))
for col in assessment_order:
    if col in pivot_data.columns:
        ax1.bar(pivot_data.index, pivot_data[col], bottom=bottom1, 
                label=col, color=colors[col])
        # Add value labels
        for i in range(len(pivot_data.index)):
            if pivot_data[col][i] > 0:  # Only show non-zero values
                ax1.text(i, bottom1[i] + pivot_data[col][i]/2,
                        f'{int(pivot_data[col][i])}',
                        ha='center', va='center')
        bottom1 += pivot_data[col]

# Plot 2: Percentages
bottom2 = np.zeros(len(pivot_pct))
for col in assessment_order:
    if col in pivot_pct.columns:
        ax2.bar(pivot_pct.index, pivot_pct[col], bottom=bottom2, 
                label=col, color=colors[col])
        # Add percentage labels
        for i in range(len(pivot_pct.index)):
            if pivot_pct[col][i] > 5:  # Only show labels for segments > 5%
                ax2.text(i, bottom2[i] + pivot_pct[col][i]/2,
                        f'{pivot_pct[col][i]:.1f}%',
                        ha='center', va='center')
        bottom2 += pivot_pct[col]

# Customize the plots
ax1.set_title('Absolute Number of Major Claims by Year', pad=20)
ax1.set_xlabel('Year Range')
ax1.set_ylabel('Number of Claims')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2.set_title('Distribution of Major Claims (%) by Year', pad=20)
ax2.set_xlabel('Year Range')
ax2.set_ylabel('Percentage of Claims')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Rotate x-axis labels for better readability
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='x', rotation=45)

# Add legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, title='Assessment Type',
          bbox_to_anchor=(1.02, 0.5), loc='center left')

# Adjust layout
plt.subplots_adjust(right=0.85, bottom=0.15)

# Print the raw numbers for reference
print("\nRaw numbers of claims by year:")
print(pivot_data)
print("\nPercentages by year:")
print(pivot_pct.round(1))

# save as png and pdf in figures/
plt.savefig('figures/claims_by_year.png', bbox_inches='tight')
plt.savefig('figures/claims_by_year.pdf', bbox_inches='tight')

# %%
claims = clean_df(claims)
claims.c

# %%
claims.columns

# %%




