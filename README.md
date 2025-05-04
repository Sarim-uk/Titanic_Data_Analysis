# Titanic Survival Prediction Project

## Project Overview
This project performs a comprehensive analysis of the Titanic dataset to predict passenger survival. The analysis includes data preprocessing, feature engineering, feature selection, and model training using machine learning techniques.

## Dataset Description
The Titanic dataset contains information about passengers aboard the RMS Titanic, which sank after colliding with an iceberg on April 15, 1912. The dataset includes various features about the passengers:

- **Survived**: Whether the passenger survived (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure
The project is organized into two main sections:

### 1. Data Preprocessing
- Loading and examining the dataset
- Handling missing values (mean imputation for Age, most frequent imputation for Embarked)
- Feature encoding (Label encoding for binary features, one-hot encoding for nominal features)
- Feature removal (dropping non-informative or redundant features like Name, Ticket, Cabin, PassengerId)

### 2. Feature Selection and Engineering
- Univariate selection using chi-squared test
- Feature importance analysis using Random Forest
- Correlation analysis with the target variable
- Creation of derived features:
  - FamilySize: Total family members on board
  - IsAlone: Whether the passenger is traveling alone
  - Age categories: Grouping passengers by age (Child, Teenager, Young Adult, Adult, Senior)

## Key Findings
- **Gender**: Being female was the strongest predictor of survival
- **Class**: Higher class passengers (1st class) had better survival rates
- **Age**: Children had higher survival rates than adults
- **Family**: Passengers traveling alone had lower survival rates
- **Embarkation**: Passengers who embarked from Cherbourg had higher survival rates

## Model Performance
The Random Forest model trained on the selected features achieved an accuracy of approximately 82%, which is a strong result for this dataset. Cross-validation was used to ensure the model's robustness.

## Technical Implementation
The project uses various Python libraries:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib** and **seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms and tools including:
  - SimpleImputer for handling missing values
  - LabelEncoder and OneHotEncoder for feature encoding
  - SelectKBest for feature selection
  - RandomForestClassifier for model training and feature importance
  - Various metrics for model evaluation

## How to Use
1. Ensure you have Python 3.x installed along with the required libraries
2. Download the Titanic dataset (train.csv) from Kaggle
3. Place the dataset in the same directory as the notebook
4. Run the notebook cells sequentially

## Future Improvements
- Experiment with additional feature engineering ideas
- Implement hyperparameter tuning for the models
- Try ensemble methods to improve prediction accuracy
- Apply more advanced preprocessing techniques
- Incorporate external data about the Titanic disaster

## Conclusion
This project demonstrates a complete machine learning workflow from data preprocessing to model evaluation. The analysis provides insights into the factors that influenced survival rates on the Titanic and shows how machine learning can be used to predict outcomes based on historical data.

The derived features and selected variables provide a strong foundation for predicting survival, and the Random Forest model performs well on this classification task. The project highlights the importance of thorough data preprocessing and feature engineering in building effective predictive models. 