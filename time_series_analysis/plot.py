import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_nvdi_max_year_and_political_violence(mine_nvdi_year_df, political_violence_year_df):
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    
    # Calculate the value counts for NVDI Max Year and normalize
    nvdi_year_counts = mine_nvdi_year_df['Year'].value_counts().sort_index()
    nvdi_year_counts_normalized = nvdi_year_counts / nvdi_year_counts.sum()
    
    # Calculate the value counts for Political Violence Year and normalize
    violence_year_counts = political_violence_year_df['Year'].value_counts().sort_index()
    violence_year_counts_normalized = violence_year_counts / violence_year_counts.sum()
    
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot the normalized counts as a line plot
    plt.plot(nvdi_year_counts_normalized.index, nvdi_year_counts_normalized.values, label='Normalized NVDI Max Year', marker='o')
    plt.plot(violence_year_counts_normalized.index, violence_year_counts_normalized.values, label='Normalized Political Violence Year', marker='o')
    
    # Add labels, title, and legend
    plt.xlabel('Year')
    plt.ylabel('Normalized Frequency')
    plt.title('Normalized Distribution of NVDI Max Year and Political Violence Year')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig('./nvdi_max_deri_political_violence_distribution.png')


if __name__ == '__main__':
    # load mine max NVDI derivative years.
    mine_nvdi_year_df = pd.read_csv('./Ghana_Mine_MaxDerivativeYear.csv')
    mine_nvdi_year_df['Year'] = mine_nvdi_year_df['Year'] + 2000
    print(mine_nvdi_year_df.head())

    # load ghana political violence years.
    political_violence_year_df = pd.read_excel('./ghana_political_violence_events_and_fatalities_by_month-year_as-of-29may2024.xlsx', sheet_name=1)
    print(political_violence_year_df.head())

    plot_nvdi_max_year_and_political_violence(mine_nvdi_year_df, political_violence_year_df)