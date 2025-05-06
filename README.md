# Titanic Dataset Analysis Project

## Project Overview
This project performs a comprehensive analysis on the Titanic dataset, focusing on data preprocessing and feature selection techniques to identify the most significant factors that influenced passenger survival on the Titanic. The project demonstrates the application of various data science methodologies to extract meaningful insights from real-world data.

## Dataset Description
The Titanic dataset is a well-known historical dataset that contains information about passengers aboard the RMS Titanic, which sank on its maiden voyage in April 1912. The dataset includes the following features:

- **Survived**: Whether the passenger survived (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure
The project is organized into two main parts:

1. **Data Preprocessing**: Cleaning, transforming, and preparing the data for analysis
2. **Feature Selection**: Identifying the most important features that influence the target variable (survival)

## Technical Implementation

### Part 1: Data Preprocessing
This section focuses on preparing the raw data for analysis by:

1. **Handling Missing Values**:
   - Imputing missing age values with the mean age
   - Imputing missing embarked values with the most frequent value

2. **Feature Dropping**:
   - Removing features that cannot be easily converted to numerical values or are not relevant for prediction:
     - Name
     - Ticket
     - Cabin
     - PassengerId

3. **Feature Encoding**:
   - Converting categorical variables to numerical form:
     - One-Hot Encoding for the 'Embarked' feature (C, Q, S)
     - Label Encoding for the 'Sex' feature (male, female)

### Part 2: Feature Selection
This section employs multiple techniques to identify the most important features:

1. **Correlation Analysis**:
   - Calculating and visualizing the correlation matrix
   - Identifying features with strong correlation to the target variable

2. **Statistical Methods**:
   - SelectKBest with ANOVA F-value to identify features with the strongest relationship to survival
   - SelectKBest with Chi-Square test as an alternative statistical approach

3. **Variance Threshold**:
   - Removing features with low variance that might not contribute significantly to prediction

4. **Principal Component Analysis (PCA)**:
   - Exploring dimensionality reduction for numerical features
   - Understanding the linear relationships between features

## Key Findings
The analysis revealed several important insights about factors affecting survival on the Titanic:

1. **Gender** was the most significant factor influencing survival, with women having a much higher survival rate
2. **Passenger Class** showed a strong negative correlation with survival, indicating that higher-class passengers had better chances of survival
3. **Fare** showed positive correlation with survival, supporting the class-based survival pattern
4. **Age** provided some predictive value, with children having better survival rates
5. **Port of Embarkation** features showed varying levels of importance

## Technical Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Usage Instructions
1. Clone this repository
2. Ensure you have the Titanic dataset (train.csv) in the project directory
3. Open and run the Jupyter notebook: `Titanic_Data_Analysis.ipynb`

## Future Work
- Implement and compare various machine learning models for prediction
- Perform feature engineering to create new features
- Conduct more advanced exploratory data analysis
- Deploy a web application for interactive visualization

## Conclusion
This project demonstrates a systematic approach to analyzing the Titanic dataset, from preprocessing raw data to identifying the key factors that influenced passenger survival. The insights gained from this analysis not only help understand historical patterns but also showcase the application of data science techniques in extracting meaningful information from complex datasets.

The combination of data preprocessing and feature selection techniques used in this project provides a solid foundation for building predictive models and can be extended to similar classification problems in other domains. 