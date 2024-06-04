# UC-Berkeley-ML-AI - Practical Assignment 3

## Overview
This project aims to build and evaluate predictive models for a marketing campaign to optimize customer subscriptions for long-term deposit products. It involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Data Description
The dataset includes both numerical and categorical features relevant to customer behavior.

### Numerical Features
- **age**: Age of the customer
- **duration**: Last contact duration in seconds
- **campaign**: Number of contacts performed during this campaign
- **pdays**: Number of days since the client was last contacted
- **previous**: Number of contacts performed before this campaign
- **emp.var.rate**: Employment variation rate
- **cons.price.idx**: Consumer price index
- **cons.conf.idx**: Consumer confidence index
- **euribor3m**: Euribor 3-month rate
- **nr.employed**: Number of employees

### Categorical Features
- **job**: Type of job
- **marital**: Marital status
- **education**: Education level
- **default**: Has credit in default?
- **housing**: Has housing loan?
- **loan**: Has personal loan?
- **contact**: Type of communication contact
- **month**: Last contact month
- **day_of_week**: Last contact day of the week
- **poutcome**: Outcome of the previous marketing campaign

## Exploratory Data Analysis (EDA)
### Key Visualizations
1. **Age Distribution**: Most customers are around 30 years old.
   ![Age Distribution]()
2. **Call Duration Distribution**: Majority of calls are short, with a few longer calls.
   ![Call Duration Distribution]()
3. **Correlation Matrix**: Highlights relationships between numerical features.
   ![Correlation Matrix]()
4. **Education vs Subscription**: Higher subscription rates for customers with university degrees.
   ![Education vs Subscription]()
5. **Job vs Subscription**: Varying subscription rates across different job types.
   ![Job vs Subscription]()
6. **Subscription Distribution**: Majority did not subscribe.
   ![Subscription Distribution]()
7. **Contact Communication Type vs Subscription**: Higher subscription rates for cellular contacts.
   ![Contact Communication Type vs Subscription]()
8. **Housing Loan vs Subscription**: Higher subscription rates for those without housing loans.
   ![Housing Loan vs Subscription]()
9. **Marital Status vs Subscription**: Single customers have higher subscription rates.
   ![Marital Status vs Subscription]()
10. **Personal Loan vs Subscription**: Higher subscription rates for those without personal loans.
    ![Personal Loan vs Subscription]()

## Model Comparison and Observing Overfitting
Four models were evaluated: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

| Model                 | Train Time (seconds) | Train Accuracy | Test Accuracy | CV Mean Accuracy | CV Std Dev |
|-----------------------|----------------------|----------------|---------------|------------------|------------|
| Logistic Regression   | 81.01                | 0.878          | 0.877         | 0.874            | 0.009      |
| Decision Tree         | 2.32                 | 0.991          | 0.960         | 0.944            | 0.002      |
| K-Nearest Neighbors   | 4.75                 | 1.000          | 0.909         | 0.913            | 0.008      |
| Support Vector Machine| 875.95               | 1.000          | 0.996         | 0.992            | 0.002      |

### Visualizations
- **Cross-Validation Mean Accuracy Comparison**
  ![CV Mean Accuracy]()
- **Test Accuracy Comparison**
  ![Test Accuracy]()
- **Training Accuracy Comparison**
  ![Training Accuracy]()
- **Training Time Comparison**
  ![Training Time]()

### Observations
- SVM shows the highest accuracy but requires more training time.
- Logistic Regression and Decision Tree models have strong performance with faster training times.
- KNN model indicates potential overfitting.

## Feature Importance Analysis
Permutation feature importance was analyzed for SVM and Logistic Regression models.

### SVM Feature Importance
Top features:
- `num_emp.var.rate`
- `num_duration`
- `num_cons.price.idx`
- `num_nr.employed`
- `num_euribor3m`

![SVM Feature Importance]()

### Logistic Regression Feature Importance
Top features:
- `num_nr.employed`
- `num_cons.conf.idx`
- `num_cons.price.idx`
- `num_euribor3m`
- `num_emp.var.rate`

![Logistic Regression Feature Importance]()

## Partial Dependence Plots (PDPs)
PDPs help understand the relationship between features and the target variable.

### Key Insights
1. **Call Duration**: Moderate durations (6-8 minutes) increase subscription likelihood.
2. **Number of Contacts During Campaign**: Fewer contacts are better.
3. **Days Since Last Contact**: More days since the last contact increases likelihood.
4. **Number of Previous Contacts**: Fewer previous contacts are better.
5. **Employment Variation Rate**: Lower rates are better.
6. **Consumer Price Index**: Moderate values are better.
7. **Consumer Confidence Index**: Moderate values are better.
8. **Month of Contact**: Specific months like March, June, September, and December might be less favorable.

![Partial Dependence Plots]()

### Business Implications
1. **Optimal Call Duration**: Maintain calls between 6 and 8 minutes.
2. **Contact Frequency**: Reduce the number of campaign contacts.
3. **Re-contact Timing**: Allow more days between contacts.
4. **Previous Contacts**: Minimize to avoid contact fatigue.
5. **Economic Indicators**: Monitor to identify favorable conditions.
6. **Month of Contact**: Adjust timing strategies based on further data validation.

## Conclusion
The SVM model is recommended for its superior performance and actionable insights. The use of permutation importance and PDPs enhances the model's interpretability, guiding strategic decisions for optimizing marketing campaigns. Further feature engineering can improve model performance and provide deeper insights into customer behavior.
