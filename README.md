# Customer Churn Prediction using Machine Learning

Customer churn prediction is a critical task for businesses aiming to retain their customer base. This project analyzes customer attrition in the telecommunications industry and employs machine learning techniques to predict churn effectively.

## Project Highlights

- **Domain**: Telecom customer retention  
- **Techniques**: Machine learning models, data preprocessing, feature selection  
- **Objective**: Predict whether a customer will leave the service based on historical data and usage patterns.

## Table of Contents

1. [Dataset Overview](#dataset-overview)  
2. [Exploratory Data Analysis](#exploratory-data-analysis)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Model Building](#model-building)  
5. [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion) 

---

## Dataset Overview

The dataset contains customer-level information, including:  

- **Demographics**: Gender, seniority, partner, dependents  
- **Account Information**: Tenure, contract type, billing methods  
- **Service Usage**: Internet services, additional features  
- **Churn Status**: Binary column indicating whether the customer left  

Dataset splits:  
- `churn-bigml-80.csv`: Training data used for model development  
- `churn-bigml-20.csv`: Testing data used for evaluation  

---
## Exploratory Data Analysis

In this phase, we conducted an in-depth analysis to understand the dataset's structure and identify patterns:

### Key Insights

1. **Churn Distribution**:  
   - Observed a class imbalance where a smaller percentage of customers churned.  

2. **Feature Correlations**:  
   - Strong correlation between tenure and churn.  
   - Monthly charges tend to vary significantly between churned and non-churned customers.  

3. **Demographic Patterns**:  
   - Customers with dependents or partners were less likely to churn.  
   - Senior citizens showed a higher churn rate.  

4. **Service Usage Trends**:  
   - Internet and additional services (e.g., streaming) influenced churn probability.  

### Visualizations

Key trends were visualized using bar plots, histograms, and boxplots to identify patterns in customer demographics, services, and churn behavior.

---

## Data Preprocessing

To prepare the data for model building, several preprocessing steps were performed:

### Steps Involved

1. **Handling Missing Values**:  
   - Checked for missing or null values in the dataset.  
   - Imputed missing values where necessary.  

2. **Encoding Categorical Variables**:  
   - Converted categorical variables (e.g., gender, contract type) into numerical format using one-hot encoding.  

3. **Feature Scaling**:  
   - Standardized numerical features like tenure and monthly charges to ensure uniform scaling.  

4. **Class Imbalance Handling**:  
   - Applied oversampling techniques such as SMOTE to balance the dataset and mitigate bias towards the majority class.

5. **Feature Selection**:  
   - Selected features with strong correlations to the t

## Model Building

We implemented multiple machine learning models to predict customer churn and compared their performance:

### Models Used

1. **Logistic Regression**  
   - A simple baseline model for binary classification tasks.  
   - Evaluated for interpretability and speed.  

2. **Random Forest Classifier**  
   - Captures complex patterns using ensemble learning.  
   - Reduced overfitting through random feature selection.  

3. **Gradient Boosting (XGBoost)**  
   - Focused on optimizing performance with boosting techniques.  
   - Demonstrated high accuracy and robustness.  

4. **Support Vector Machine (SVM)**  
   - Applied with a radial basis function (RBF) kernel for better classification in non-linear spaces.  

### Hyperparameter Tuning

Performed hyperparameter optimization using GridSearchCV to identify the best parameters for:  
- Maximum tree depth and number of estimators for Random Forest.  
- Learning rate and tree depth for XGBoost.  
- Regularization parameters for SVM.  

### Output

Each model was trained on the preprocessed training dataset, and predictions were generated for the test dataset.

---

## Model Evaluation

The performance of each model was evaluated using several metrics to ensure reliable and interpretable results:

### Evaluation Metrics
1. **Accuracy**:  
   - Measures the proportion of correct predictions among the total predictions.

2. **Precision and Recall**:  
   - Precision: Proportion of true positive predictions among all positive predictions.  
   - Recall: Ability to identify all positive instances correctly.  

3. **F1 Score**:  
   - Harmonic mean of precision and recall to balance the trade-off.  

4. **ROC-AUC Score**:  
   - Assesses the model's ability to distinguish between classes.  

### Key Results
- **Random Forest**: Achieved the highest accuracy and recall, showing robustness to class imbalance.  
- **XGBoost**: Delivered competitive performance with slightly better precision.  
- **Logistic Regression**: Served as a strong baseline but was outperformed by ensemble models.  
- **SVM**: Struggled with scalability due to dataset size but provided decent results for balanced data.  

---

## Conclusion

This project successfully predicts customer churn using machine learning techniques, offering valuable insights to businesses for proactive decision-making. 

### Key Takeaways
- Preprocessing and feature engineering significantly improved model performance.  
- Random Forest proved to be the most effective model, balancing accuracy, interpretability, and robustness.  

### Future Scope
- Incorporate additional data features, such as customer feedback and interaction logs, for improved prediction accuracy.  
- Explore deep learning models for better handling of complex patterns in customer behavior.  
- Extend the deployment to cloud platforms like AWS or Azure for scalability and accessibility.  

This project demonstrates the potential of data-driven solutions in addressing critical business challenges like customer retention.

---
