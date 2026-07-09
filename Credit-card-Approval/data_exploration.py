import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
app_df = pd.read_csv('data/application_record.csv')
credit_df = pd.read_csv('data/credit_record.csv')

# Quick look at the data
print("Application Record Shape:", app_df.shape)
print("Credit Record Shape:", credit_df.shape)
print("\nApplication Record Head:")
print(app_df.head())
print("\nCredit Record Head:")
print(credit_df.head())
# Analyze credit status
print("\nCredit STATUS unique values:")
print(credit_df['STATUS'].unique())

print("\nCredit STATUS value counts:")
print(credit_df['STATUS'].value_counts())

# Check how many unique clients
print(f"\nUnique clients in credit records: {credit_df['ID'].nunique()}")
print(f"Unique clients in application records: {app_df['ID'].nunique()}")

# Define function to classify clients as good or bad
def classify_client(status_history):
    """
    Classify client as:
    - 'bad' if they have any status in ['1','2','3','4','5'] (past due 30+ days)
    - 'good' otherwise
    """
    for status in status_history:
        if status in ['1','2','3','4','5']:
            return 'bad'
    return 'good'

# Group by ID and apply classification
credit_grouped = credit_df.groupby('ID')['STATUS'].apply(list).reset_index()
credit_grouped['TARGET'] = credit_grouped['STATUS'].apply(classify_client)

print("\nTarget distribution:")
print(credit_grouped['TARGET'].value_counts())
print(f"\nPercentage: \n{credit_grouped['TARGET'].value_counts(normalize=True) * 100}")
# Merge application data with target labels
merged_df = app_df.merge(credit_grouped[['ID', 'TARGET']], on='ID', how='inner')

print(f"\nMerged dataset shape: {merged_df.shape}")
print(f"\nMerged target distribution:")
print(merged_df['TARGET'].value_counts())
print(f"\nPercentage: \n{merged_df['TARGET'].value_counts(normalize=True) * 100}")

# Check for missing values
print(f"\nMissing values per column:")
print(merged_df.isnull().sum())

# Handle missing values in OCCUPATION_TYPE
merged_df['OCCUPATION_TYPE'] = merged_df['OCCUPATION_TYPE'].fillna('Unknown')

# Create age feature from DAYS_BIRTH (negative values, convert to positive years)
merged_df['AGE'] = abs(merged_df['DAYS_BIRTH']) // 365

# Create employment years feature
merged_df['EMPLOYMENT_YEARS'] = merged_df['DAYS_EMPLOYED'].apply(lambda x: abs(x) // 365 if x < 0 else 0)

# Check the new features
print("\nAge statistics:")
print(merged_df['AGE'].describe())

print("\nEmployment years statistics:")
print(merged_df['EMPLOYMENT_YEARS'].describe())

print("\nMissing values after handling:")
print(merged_df.isnull().sum())

# Set up visualization style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Plot 1: Target distribution
plt.subplot(2, 2, 1)
target_counts = merged_df['TARGET'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.bar(target_counts.index, target_counts.values, color=colors)
plt.title('Target Distribution (Good vs Bad)', fontsize=14)
plt.xlabel('Client Type')
plt.ylabel('Count')
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')

# Plot 2: Age distribution by target
plt.subplot(2, 2, 2)
sns.histplot(data=merged_df, x='AGE', hue='TARGET', bins=30, alpha=0.6)
plt.title('Age Distribution by Client Type', fontsize=14)

# Plot 3: Income distribution by target
plt.subplot(2, 2, 3)
sns.boxplot(data=merged_df, x='TARGET', y='AMT_INCOME_TOTAL')
plt.title('Income Distribution by Client Type', fontsize=14)
plt.yscale('log')

# Plot 4: Employment years by target
plt.subplot(2, 2, 4)
sns.boxplot(data=merged_df, x='TARGET', y='EMPLOYMENT_YEARS')
plt.title('Employment Years by Client Type', fontsize=14)

plt.tight_layout()
plt.show()

# Check for outliers in numerical columns
numerical_cols = ['AMT_INCOME_TOTAL', 'AGE', 'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']

print("\n=== OUTLIER ANALYSIS ===")
for col in numerical_cols:
    Q1 = merged_df[col].quantile(0.25)
    Q3 = merged_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = merged_df[(merged_df[col] < lower_bound) | (merged_df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(merged_df)*100:.2f}%)")

# Box plots for outlier visualization
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=merged_df, x='TARGET', y=col)
    plt.title(f'{col} by Target')
    plt.yscale('log' if col == 'AMT_INCOME_TOTAL' else 'linear')
plt.tight_layout()
plt.show()

# Check extreme income values
print("\n=== TOP 10 HIGHEST INCOMES ===")
print(merged_df.nlargest(10, 'AMT_INCOME_TOTAL')[['ID', 'AMT_INCOME_TOTAL', 'TARGET', 'AGE', 'EMPLOYMENT_YEARS']])

# Check class imbalance more detail
print("\n=== CLASS IMBALANCE DETAIL ===")
print(f"Total samples: {len(merged_df)}")
print(f"Good clients: {len(merged_df[merged_df['TARGET']=='good'])} ({len(merged_df[merged_df['TARGET']=='good'])/len(merged_df)*100:.2f}%)")
print(f"Bad clients: {len(merged_df[merged_df['TARGET']=='bad'])} ({len(merged_df[merged_df['TARGET']=='bad'])/len(merged_df)*100:.2f}%)")
print(f"Imbalance ratio: {len(merged_df[merged_df['TARGET']=='good'])/len(merged_df[merged_df['TARGET']=='bad']):.2f}:1")

# Handle outliers using capping (winsorization)
def cap_outliers(df, col, lower_percentile=0.01, upper_percentile=0.99):
    lower = df[col].quantile(lower_percentile)
    upper = df[col].quantile(upper_percentile)
    df[col] = df[col].clip(lower, upper)
    return df

# Apply capping to numerical columns
numerical_cols_to_cap = ['AMT_INCOME_TOTAL', 'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']

for col in numerical_cols_to_cap:
    merged_df = cap_outliers(merged_df, col)
    print(f"Capped {col} at 1st and 99th percentiles")

# Verify outliers are handled
print("\n=== AFTER CAPPING: OUTLIER CHECK ===")
for col in numerical_cols_to_cap:
    Q1 = merged_df[col].quantile(0.25)
    Q3 = merged_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = merged_df[(merged_df[col] < lower_bound) | (merged_df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(merged_df)*100:.2f}%)")

# Save the cleaned data for modeling
merged_df.to_csv('data/cleaned_data.csv', index=False)
print("\n Cleaned data saved to 'data/cleaned_data.csv'")
# Better outlier handling using IQR method
def handle_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower_bound, upper_bound)
    return df

# Apply IQR capping
for col in ['AMT_INCOME_TOTAL', 'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']:
    merged_df = handle_outliers_iqr(merged_df, col)
    print(f"Capped {col} using IQR method")

# Verify outliers are handled
print("\n=== AFTER IQR CAPPING: OUTLIER CHECK ===")
for col in ['AMT_INCOME_TOTAL', 'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']:
    Q1 = merged_df[col].quantile(0.25)
    Q3 = merged_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = merged_df[(merged_df[col] < lower_bound) | (merged_df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(merged_df)*100:.2f}%)")

# Save the properly cleaned data
merged_df.to_csv('data/cleaned_data_final.csv', index=False)
print("\nFinal cleaned data saved to 'data/cleaned_data_final.csv'")