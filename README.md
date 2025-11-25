# AI_ML_task_5


# Heart Disease Prediction with Decision Trees and Random Forests

This project demonstrates how to use **Decision Trees** and **Random Forests** to predict heart disease using a dataset from UCI. It covers the entire process of building, visualizing, and evaluating machine learning models, with a focus on interpretability and performance. The final result is a production-ready analysis with insightful visualizations that are both informative and stunning.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Decision Tree Model](#decision-tree-model)
- [Random Forest Model](#random-forest-model)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Visualizations](#visualizations)
- [Final Insights](#final-insights)

## Introduction

In this project, the goal is to predict whether a patient has heart disease based on medical attributes. The dataset comes from the **UCI Heart Disease Database**, which includes various medical features such as age, cholesterol levels, and resting blood pressure.

We use two powerful machine learning techniques:
1. **Decision Trees** – A simple yet interpretable model.
2. **Random Forests** – An ensemble model that combines the predictions of many decision trees for improved accuracy.

We'll also analyze model overfitting, perform hyperparameter tuning, and evaluate the final model using key metrics such as accuracy, precision, recall, and AUC.

## Dataset Overview

We used the **Heart Disease dataset** from UCI, which includes 303 samples with 14 features (including age, sex, cholesterol, and others). The target variable indicates whether a patient has heart disease (1) or not (0).

The dataset was loaded using the `fetch_openml` function and cleaned for better usability.

## Data Preprocessing

1. **Renaming Columns**: We gave the columns more descriptive names for clarity.
2. **Target Mapping**: The target variable, which originally had "present" or "absent" labels, was mapped to 1 (present) and 0 (absent).
3. **Categorical Features**: We converted the categorical variables (like sex, chest pain, etc.) into a proper data type for processing.
4. **Train-Test Split**: The dataset was split into a training set (80%) and a testing set (20%) for model evaluation.

## Decision Tree Model

### Full Tree vs. Pruned Tree

- **Full Tree**: We first built a decision tree without constraints, which often results in overfitting. This tree captures every possible pattern in the training data, but it doesn't generalize well to unseen data.
- **Pruned Tree**: To prevent overfitting, we pruned the tree by limiting its depth and requiring at least 5 samples in each leaf node. This results in a simpler model that generalizes better.

### Model Performance

We compared the accuracy of the full tree with the pruned tree. The pruned tree performed better, showing the importance of controlling model complexity.

### Visualization

The pruned decision tree was visualized to show how the model makes decisions based on the different features. We also exported the tree as a `.dot` file for high-resolution rendering using Graphviz.

## Random Forest Model

### Ensemble Learning

We built a **Random Forest** with 500 trees, a technique that uses multiple decision trees to make more accurate predictions. The Random Forest was trained with:
- **Bootstrap sampling** (sampling with replacement).
- **Out-of-bag (OOB) score** for internal cross-validation.

### Hyperparameter Tuning

Using **GridSearchCV**, we tuned the Random Forest model by trying different values for the number of trees (`n_estimators`), tree depth (`max_depth`), and minimum samples required for splitting or leaf nodes. The best parameters were selected based on cross-validation performance.

## Model Evaluation

### Key Metrics
We evaluated the models based on:
- **Accuracy**: How often the model correctly predicted the target.
- **Precision**: The proportion of true positives out of all predicted positives.
- **Recall**: The proportion of true positives out of all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: The area under the receiver operating characteristic curve, showing the trade-off between sensitivity and specificity.

### Results

The **Random Forest** model outperformed the decision trees in all metrics, especially in **ROC-AUC**, demonstrating its ability to handle the complexity of the problem effectively.

## Feature Importance

One of the key strengths of the Random Forest model is its ability to rank features by their importance in predicting the target. We extracted the top 10 most important features for heart disease prediction, which can help clinicians focus on the most relevant factors.

## Visualizations

We created several high-quality visualizations to aid understanding and communication of the model's performance:

1. **Feature Importance**: A bar plot showing the most important features in the Random Forest model.
2. **ROC Curve**: A comparison of the ROC curves for the pruned decision tree, full decision tree, and Random Forest model.
3. **Confusion Matrix**: A heatmap showing the confusion matrix of the Random Forest model.
4. **Learning Curve**: A plot that shows how the model’s accuracy improves with increasing training data.
5. **OOB Error vs. Ensemble Size**: A plot showing the out-of-bag error rate as a function of the number of trees in the Random Forest.

## Final Insights

The Random Forest model was found to be the best performer, with an **AUC of X.XXXX**. The top three features driving the prediction were:
- `Feature 1`
- `Feature 2`
- `Feature 3`

### Takeaways:
- **Decision Trees** are interpretable but prone to overfitting without proper tuning.
- **Random Forests** provide robust performance by aggregating the predictions of multiple trees.
- **Feature importance** in Random Forests provides actionable insights for domain experts (e.g., medical professionals).
- The **OOB score** provides a quick estimate of model performance without needing a separate validation set.

## Running the Code

To run this project on your local machine, you need the following dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `graphviz`

You can install them using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn graphviz
