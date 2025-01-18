import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "unemployment_data.csv"  # Replace with your dataset path
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Handle missing values if necessary
df.fillna(method='ffill', inplace=True)

# Convert date column to datetime format (if applicable)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Analyze unemployment trends over time
if 'Unemployment Rate' in df.columns and 'Date' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Unemployment Rate', marker='o')
    plt.title('Unemployment Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.grid()
    plt.show()

# Analyze unemployment by category (e.g., by state or industry)
if 'State' in df.columns and 'Unemployment Rate' in df.columns:
    avg_unemployment_by_state = df.groupby('State')['Unemployment Rate'].mean().sort_values(ascending=False)
    print("\nAverage Unemployment Rate by State:")
    print(avg_unemployment_by_state)

    # Plot unemployment rate by state
    plt.figure(figsize=(12, 8))
    avg_unemployment_by_state.plot(kind='bar', color='skyblue')
    plt.title('Average Unemployment Rate by State')
    plt.xlabel('State')
    plt.ylabel('Unemployment Rate (%)')
    plt.xticks(rotation=45)
    plt.show()

# Correlation analysis (if there are numerical features)
if df.select_dtypes(include=['float64', 'int64']).shape[1] > 1:
    correlation_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Heatmap of correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
