import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Import Dataset
# ==========================================

df = pd.read_csv("StudentsPerformance.csv")

# ==========================================
# Basic Dataset Information
# ==========================================

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# Subject Columns
# ==========================================

subjects = ['math score', 'reading score', 'writing score']

# ==========================================
# Average Subject Scores
# ==========================================

avg_scores = df[subjects].mean()

print("\nAverage Subject Scores:")
print(avg_scores)

# ==========================================
# Bar Chart — Average Subject Scores
# ==========================================

plt.figure(figsize=(8,5))

avg_scores.plot(kind='bar')

plt.title("Average Subject Scores")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)

plt.show()

# ==========================================
# Histogram — Distribution of Math Scores
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df['math score'],
    bins=20,
    kde=True
)

plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")

plt.show()

# ==========================================
# Gender Comparison Analysis
# ==========================================

gender_scores = df.groupby('gender')[subjects].mean()

print("\nGender-wise Average Scores:")
print(gender_scores)

# ==========================================
# Gender vs Subject Scores (Bar Chart)
# ==========================================

gender_scores.plot(kind='bar', figsize=(8,5))

plt.title("Gender vs Subject Scores")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)

plt.show()

# ==========================================
# Test Preparation Analysis
# ==========================================

prep_scores = df.groupby(
    'test preparation course'
)[subjects].mean()

print("\nTest Preparation Impact:")
print(prep_scores)

# ==========================================
# Test Preparation Impact (Bar Chart)
# ==========================================

prep_scores.plot(kind='bar', figsize=(8,5))

plt.title("Test Preparation Impact")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)

plt.show()

# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df[subjects].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Between Subjects")

plt.show()

# ==========================================
# Overall Student Performance
# ==========================================

df['average score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
) / 3

print("\nAverage Score Statistics:")
print(df['average score'].describe())

# ==========================================
# Top Performing Students
# ==========================================

top_students = df.sort_values(
    by='average score',
    ascending=False
).head(10)

print("\nTop Performing Students:")
print(top_students)

# ==========================================
# Summary
# ==========================================

print("\nSummary:")

print("1. Reading and Writing scores are highly correlated.")

print("2. Students completing test preparation perform better.")

print("3. Female students perform better in reading and writing.")

print("4. Average subject scores are nearly similar.")

print("\nProjct Completed Successfully.\n")