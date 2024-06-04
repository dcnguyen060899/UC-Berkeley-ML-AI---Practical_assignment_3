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
1. **Age Distribution**:
   ![Age Distribution](age_distribution.png)
   - Most customers are around 30 years old.
   - The age distribution shows a right skew, indicating fewer older customers.

2. **Call Duration Distribution**:
   ![Call Duration Distribution](call_duration_distribution.png)
   - Majority of calls are short, typically under 500 seconds.
   - The distribution is heavily right-skewed, with a long tail of longer calls.

3. **Correlation Matrix**:
   ![Correlation Matrix](correlation_matrix.png)
   - Highlights positive and negative relationships between numerical features.
   - `duration` and `y_encoded` have a moderate positive correlation, suggesting longer calls are associated with higher subscription rates.

4. **Education vs Subscription**:
   ![Education vs Subscription](education_subscription_distribution.png)
   - Higher subscription rates are observed among customers with university degrees.
   - Customers with lower educational levels tend to have lower subscription rates.

5. **Job vs Subscription**:
   ![Job vs Subscription](job_subscription_distribution.png)
   - Job types like 'admin.' and 'blue-collar' have lower subscription rates.
   - Higher subscription rates are observed for job types like 'management' and 'retired'.

6. **Subscription Distribution**:
   ![Subscription Distribution](subscription_distribution.png)
   - Majority of the customers did not subscribe.
   - There is a significant class imbalance with far fewer subscribers.

7. **Contact Communication Type vs Subscription**:
   ![Contact Communication Type vs Subscription](contact_communication_type_subscription_distribution.png)
   - Higher subscription rates are seen for cellular contacts compared to telephone.
   - The majority of contacts were made via cellular phones.

8. **Housing Loan vs Subscription**:
   ![Housing Loan vs Subscription](housing_loan_subscription_distribution.png)
   - Customers without housing loans have higher subscription rates.
   - Most customers, irrespective of subscription status, do have housing loans.

9. **Marital Status vs Subscription**:
   ![Marital Status vs Subscription](martial_status_subscription_distribution.png)
   - Single customers have higher subscription rates compared to married or divorced ones.
   - Married customers form the largest group but have lower subscription rates.

10. **Personal Loan vs Subscription**:
    ![Personal Loan vs Subscription](personal_loan_subscription_distribution.png)
    - Higher subscription rates are observed for customers without personal loans.
    - Most customers, especially those who did not subscribe, do not have personal loans.

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
  ![CV Mean Accuracy](cross_validate_mean_accuracy_comparison.png)
- **Test Accuracy Comparison**
  ![Test Accuracy](test_accuracy_comparison.png)
- **Training Accuracy Comparison**
  ![Training Accuracy](training_accuracy_comparison.png)
- **Training Time Comparison**
  ![Training Time](training_time_comparison.png)

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

![SVM Feature Importance](permutation_feature_importance_svm.png)

### Logistic Regression Feature Importance
Top features:
- `num_nr.employed`
- `num_cons.conf.idx`
- `num_cons.price.idx`
- `num_euribor3m`
- `num_emp.var.rate`

![Logistic Regression Feature Importance](permutation_feature_importance_lr.png)

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

![Partial Dependence Plots](pdpx.png)

### Business Implications
1. **Optimal Call Duration**: Maintain calls between 6 and 8 minutes.
2. **Contact Frequency**: Reduce the number of campaign contacts.
3. **Re-contact Timing**: Allow more days between contacts.
4. **Previous Contacts**: Minimize to avoid contact fatigue.
5. **Economic Indicators**: Monitor to identify favorable conditions.
6. **Month of Contact**: Adjust timing strategies based on further data validation.

## Conclusion
The SVM model is recommended for its superior performance and actionable insights. The use of permutation importance and PDPs enhances the model's interpretability, guiding strategic decisions for optimizing marketing campaigns. Further feature engineering can improve model performance and provide deeper insights into customer behavior.
