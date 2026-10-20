# File Name: titanic_dataviz.py
# Description: Task 3 - Data Visualization Dashboard for CodeAlpha

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("==========================================")
print("TASK 3: DATA VISUALIZATION DASHBOARD")
print("==========================================")

# 1. Load & Prepare Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Set global plotting theme
sns.set_theme(style="whitegrid")

# 2. Construct 2x2 Subplot Dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Titanic Passenger Survival Analysis Dashboard', fontsize=16, fontweight='bold', y=0.95)

# Chart 1: Overall Survival Count
sns.countplot(x='Survived', data=df, ax=axes[0, 0], palette='Set2', hue='Survived', legend=False)
axes[0, 0].set_title('Overall Survival Count (0 = Died, 1 = Survived)', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Died', 'Survived'])
axes[0, 0].set_xlabel('Survival Status')
axes[0, 0].set_ylabel('Passenger Count')

# Chart 2: Survival Probability by Gender
sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0, 1], palette='Set1', hue='Sex', errorbar=None, legend=False)
axes[0, 1].set_title('Survival Rate Probability by Gender', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Survival Probability')

# Chart 3: Survival Rate by Passenger Class & Gender
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ax=axes[1, 0], palette='Blues_d', errorbar=None)
axes[1, 0].set_title('Survival Rate by Passenger Class & Gender', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
axes[1, 0].set_ylabel('Survival Probability')
axes[1, 0].legend(title='Gender')

# Chart 4: Age Distribution (Survivors vs Non-Survivors)
df_vis = df.copy()
df_vis['Survival_Status'] = df_vis['Survived'].map({0: 'Died', 1: 'Survived'})
sns.histplot(
    data=df_vis, 
    x='Age', 
    hue='Survival_Status', 
    kde=True, 
    ax=axes[1, 1], 
    palette={'Died': '#e74c3c', 'Survived': '#2ecc71'}, 
    bins=30, 
    alpha=0.6
)
axes[1, 1].set_title('Age Distribution: Survivors vs Non-Survivors', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Age (Years)')
axes[1, 1].set_ylabel('Density / Count')
axes[1, 1].legend(title='Outcome', loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
print("\nRendering visualization dashboard window...")
plt.show()