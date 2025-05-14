# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows
print("First 5 rows of the dataset:\n", df.head())

# Data info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Basic statistics
print("\nBasic Statistics:\n", df.describe())

# Group by species
grouped_means = df.groupby('species').mean()
print("\nGrouped Means by Species:\n", grouped_means)

# Set seaborn style
sns.set(style="whitegrid")

# Plot 1: Line chart (simulated time)
df['index'] = df.index
plt.figure()
plt.plot(df['index'], df['sepal length (cm)'])
plt.title("Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.savefig("line_chart.png")

# Plot 2: Bar chart
grouped_means['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.clf()

# Plot 3: Histogram
plt.hist(df['sepal width (cm)'], bins=15, color='lightgreen')
plt.title("Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.clf()

# Plot 4: Scatter plot
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")

print("\n✅ Analysis and plots complete. Images saved in the project folder.")
