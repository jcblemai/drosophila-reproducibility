# Companion repository for _A retrospective analysis of 400 publications reveals patterns of irreproducibility across an entire life sciences research field_

## Overview

This repository contains the complete analysis code and data for a comprehensive study examining reproducibility patterns across 400 publications in Drosophila research. It is used to generates all figure (see example below) and all statistical analysis.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/416619e7-a2c2-41cf-9384-cf483a747708" />

<img width="300" alt="image" src="https://github.com/user-attachments/assets/e1f3297f-3d6c-40dd-af08-dab6e50c7d8c" />

<img width="300" alt="image" src="https://github.com/user-attachments/assets/c0a30fc9-8976-4968-b0a6-0dd7b5571f62" />

## Repository Structure

```
├── preprocessed_data/          # Processed datasets ready for analysis
├── analysis_claims.py         # Main claim analysis (Figures 1-3)
├── analysis_authors_first.py  # First author analysis (Figures 4-5)
├── analysis_authors_last.py   # Last author analysis (Figures 6-7, 9)
├── statistical_analysis.py    # Multivariate model (Figure 10)
├── plot_info.py              # Plotting utilities
├── wrangling.py              # Data processing functions
├── stat_lib.py               # Statistical analysis functions
├── preprocess_db.ipynb       # Database preprocessing
└── preprocess_xlsx.ipynb     # Excel file preprocessing
```

## Quick Start

### Option 1: Using Preprocessed Data (Recommended)

The simplest way to reproduce all figures and analyses is to use the preprocessed data stored in `preprocessed_data/`:

```bash
# Run the main analyses
python analysis_claims.py        # Generates Main Text Figures 1-3
python analysis_authors_first.py # Generates Main Text Figures 4-5
python analysis_authors_last.py  # Generates Main Text Figures 6-7, 9
python statistical_analysis.py   # Generates Main Text Figure 10
```
This generates all paper figures, all tables, all numbers in the text with Wilson confidence interval, all the ddd-ratios for categorical variable comparison with significance test, and the main model, a random effect mixed regression (with Bambi) with diagnostic checks (rhat, posterior predictive checks) ... 

### Option 2: Full Preprocessing Pipeline

To reproduce the complete preprocessing pipeline:

1. Download the SQL dump from the ReproSci database (https://reprosci.epfl.ch)
2. Run `preprocess_db.ipynb` to process the database
3. Run `preprocess_xlsx.ipynb` to extract manual covariates from Excel files
4. Execute the analysis scripts as above

## Data Files store in the repository.

### Core Datasets (`processed_data/`)

- **`article_db.csv`**: Article metadata (journal, year) from ReproSci database
- **`author_db.csv`**: Author information (sex, etc.) from ReproSci database
- **`claims_db_truncated.csv`**: Main dataset with one row per claim, merged with article and author data
- **`first_author_claims.csv`**: First author covariates merged with claim data
- **`last_author_claims.csv`**: Last author covariates merged with claim data

## Data Sources
The analysis uses data from the ReproSci database (https://reprosci.epfl.ch). It is built from an SQL dump but you can also query it using:
- Claims database: https://reprosci.epfl.ch/workspaces/3tebs5/get_claims
- Author statistics: https://reprosci.epfl.ch/download?f=author_stats
- Challenge reasons: https://reprosci.epfl.ch/reasons
- Citation counts: https://reprosci.epfl.ch/download?f=citation_counts

## Dependencies
- PyMC (for Bayesian modeling)
- Bambi for model buiding
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scipy (statistical tests)

