import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

# Load dataset from CSV file
csv_path = 'data.csv'
laptop_df = pd.read_csv(csv_path)

# Data Preprocessing
# Removing unnecessary column
laptop_df.drop(columns=['Unnamed: 16'], inplace=True)

# Cleaning 'Ram' column: Removing 'GB' suffix and converting to integer
laptop_df['Ram'] = laptop_df['Ram'].str.replace('GB', '').astype(int)

# Cleaning 'Cpu Rate' column: Removing 'GHz' suffix and converting to float
laptop_df['Cpu Rate'] = laptop_df['Cpu Rate'].str.replace('GHz', '').astype(float)

# Handling missing values in storage-related columns
storage_cols = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']
laptop_df[storage_cols] = laptop_df[storage_cols].fillna(0).astype(int)

# Custom color palette for visualization
color_palette = ['#004c6d', '#ff6b35', '#7aa874', '#ee4266', '#3a506b', '#f4a261', '#ffb703', '#8d99ae', '#e63946', '#06d6a0']

def plot_price_distribution(data):
    """
    Displays histogram with KDE for laptop price distribution.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Price_euros'], bins=30, kde=True, color="#004c6d", edgecolor="black")
    plt.title('Laptop Price Distribution', fontweight='bold', fontsize=14)
    plt.xlabel('Price in Euros', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_price_by_type(data):
    """
    Boxplot representing laptop prices grouped by type.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='TypeName', y='Price_euros', data=data, palette=color_palette)
    plt.title('Laptop Price by Type', fontweight='bold', fontsize=14)
    plt.xlabel('Laptop Type', fontweight='bold')
    plt.ylabel('Price in Euros', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_price_vs_screen_size(data):
    """
    Line plot showing trend of laptop price based on screen size.
    """
    plt.figure(figsize=(10, 6))
    screen_price_avg = data.groupby('Inches')['Price_euros'].mean().reset_index()
    sns.lineplot(x='Inches', y='Price_euros', data=screen_price_avg, marker='o', color=color_palette[2])
    plt.title('Trend of Laptop Price vs. Screen Size', fontweight='bold', fontsize=14)
    plt.xlabel('Screen Size (Inches)', fontweight='bold')
    plt.ylabel('Average Price in Euros', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_avg_price_by_ram(data):
    """
    Bar chart displaying average laptop price categorized by RAM size.
    """
    plt.figure(figsize=(10, 6))
    ram_avg_price = data.groupby('Ram')['Price_euros'].mean().reset_index()
    sns.barplot(x='Ram', y='Price_euros', data=ram_avg_price, palette=color_palette)
    plt.title('Average Laptop Price by RAM Size', fontweight='bold', fontsize=14)
    plt.xlabel('RAM (GB)', fontweight='bold')
    plt.ylabel('Average Price in Euros', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_avg_price_by_brand(data):
    """
    Bar chart showing the average laptop price grouped by brand.
    """
    plt.figure(figsize=(12, 8))
    brand_avg_price = data.groupby('Company')['Price_euros'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='Price_euros', y='Company', data=brand_avg_price, palette=color_palette)
    plt.title('Average Laptop Price by Brand', fontweight='bold', fontsize=14)
    plt.xlabel('Average Price (Euros)', fontweight='bold')
    plt.ylabel('Brand', fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()

# Statistical Summary
stats_summary = laptop_df.describe()

# Correlation analysis
correlation_data = laptop_df.corr(numeric_only=True)

# Kurtosis and Skewness assessment for price distribution
price_kurt = kurtosis(laptop_df['Price_euros'], fisher=True)
price_skew = skew(laptop_df['Price_euros'])

# Display statistical insights
print("Descriptive Statistics:\n", stats_summary)
print("\nCorrelation Matrix:\n", correlation_data)
print("\nPrice Kurtosis:", price_kurt)
print("Price Skewness:", price_skew)

# Generate visualizations
plot_price_distribution(laptop_df)
plot_price_by_type(laptop_df)
plot_price_vs_screen_size(laptop_df)
plot_avg_price_by_ram(laptop_df)
plot_avg_price_by_brand(laptop_df)
