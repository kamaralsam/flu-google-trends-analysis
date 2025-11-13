**Google Trends vs CDC Flu Data Analysis**

A Python data project exploring whether Google search behavior can predict flu outbreaks.

**📌 Project Overview**

This project analyzes the relationship between Google Trends search activity for flu-related keywords and CDC Influenza-Like Illness (ILI) data. The goal is to evaluate whether Google Trends can serve as an early “digital surveillance” indicator for flu outbreaks.

After encountering API rate limits, I redesigned the workflow to use local CSV ingestion, creating a stable, fully reproducible pipeline.

**🔧 Tools & Technologies**

Python

Pandas, NumPy, SciPy

Matplotlib, Seaborn

Time-series alignment & correlation analysis

Custom DataCollector class

Local CSV ingestion (no API calls)

**🔄 Workflow Summary**
**1. Data Collection (Local CSVs)**

Created a DataCollector class to load:

Google Trends CSV

CDC ILI CSV

Automatically validates paths, handles formatting differences, and converts dates.

**2. Data Cleaning & Preprocessing**

Standardized date formats

Filled missing values / missing weeks

Resampled weekly data

Aligned the two time series for analysis

**3. Exploratory Data Analysis**

Generated:

Overlaid time-series plots

Scatterplots for correlation

Lag comparison plots

**4. Statistical Analysis**

Pearson correlation between search trends and CDC data

Lag analysis to test whether Google Trends spikes appear earlier than CDC flu peaks

**📊 Key Findings**

Strong positive correlation between Google search interest and CDC ILI rates

Google Trends often peaks 1–2 weeks earlier than CDC reports

Supports the idea that search behavior may signal flu outbreaks early

**📁 Project Structure**
project-folder/
│── data/
│    ├── google_trends.csv
│    ├── cdc_ili.csv
│── src/
│    ├── data_collector.py
│    ├── analysis.py
│    ├── visualizations.py
│── notebooks/
│    └── flu_analysis.ipynb
│── README.md

**Result**

A reproducible, stable data analytics project that integrates

1) time-series data cleaning
2) correlation & lag analysis
3) statistical reasoning
4) Python visualization
5) digital epidemiology concepts
