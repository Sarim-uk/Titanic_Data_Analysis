import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Titanic Dataset Analysis: Data Pre-Processing and Feature Selection"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Importing Titanic Dataset"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import seaborn as sns\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n",
                "from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "\n",
                "df = pd.read_csv('train.csv')\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Part 1: Data Pre-Processing"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Handle missing values\n",
                "imputer = SimpleImputer(strategy='mean')\n",
                "df['Age'] = imputer.fit_transform(df[['Age']]).flatten()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Drop unnecessary features\n",
                "df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])\n",
                "df"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Handle Embarked\n",
                "imputer = SimpleImputer(strategy='most_frequent')\n",
                "df['Embarked'] = imputer.fit_transform(df[['Embarked']]).flatten()\n",
                "\n",
                "# One-hot encode Embarked\n",
                "encoder = OneHotEncoder(sparse_output=False)\n",
                "enc = encoder.fit_transform(df[['Embarked']])\n",
                "df_encoded = pd.DataFrame(enc, columns=encoder.categories_[0])\n",
                "df = pd.concat([df, df_encoded], axis=1)\n",
                "df = df.drop(columns=['Embarked'])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Label encode Sex\n",
                "label_encoder = LabelEncoder()\n",
                "df['Sex'] = label_encoder.fit_transform(df['Sex'])\n",
                "df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Part 2: Feature Selection"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Correlation analysis\n",
                "correlation_matrix = df.corr()\n",
                "plt.figure(figsize=(12, 8))\n",
                "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')\n",
                "plt.title('Correlation Matrix of Features')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ANOVA F-value\n",
                "X = df.drop('Survived', axis=1)\n",
                "y = df['Survived']\n",
                "\n",
                "selector = SelectKBest(score_func=f_classif, k=5)\n",
                "X_new = selector.fit_transform(X, y)\n",
                "selected_features = X.columns[selector.get_support(indices=True)]\n",
                "print(\"Top features selected by ANOVA F-value:\")\n",
                "print(selected_features)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# PCA\n",
                "df_pca = df[['Fare', 'Age']]\n",
                "scaler = StandardScaler()\n",
                "df_pca_scaled = scaler.fit_transform(df_pca)\n",
                "pca = PCA(0.95)\n",
                "df_pca_transformed = pca.fit_transform(df_pca_scaled)\n",
                "print(\"PCA transformed data shape:\", df_pca_transformed.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "After comprehensive data preprocessing and feature selection analysis on the Titanic dataset, we've identified the most significant features that influence survival prediction:\n",
                "\n",
                "1. Sex is the most strongly correlated feature with survival\n",
                "2. Pclass (passenger class) shows a strong negative correlation\n",
                "3. Fare shows positive correlation with survival\n",
                "4. Age provides some predictive value\n",
                "5. The port of embarkation features (C, Q, S) show varying levels of importance"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Write the notebook to file
with open('Titanic_Data_Analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully.") 