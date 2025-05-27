
# Student Performance Classification using Machine Learning

This project aims to predict and classify students' final performance in a mathematics course using machine learning models. The prediction categorizes students into three performance classes: **Fail**, **Pass**, and **Excellent** based on demographic, academic, and social attributes.

## Dataset

- **Name**: `student-mat.csv`
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **Attributes**: 33 features including grades, age, study time, family background, and support systems
- **Target Variable**: Final grade `G3`, categorized as:
  - `0` = Fail (0–9)
  - `1` = Pass (10–14)
  - `2` = Excellent (15–20)

## Objective

Build and compare classification models to:
- Accurately classify students into performance categories
- Analyze feature importance
- Improve interpretability and generalization

## Tools & Libraries

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Google Colab (execution environment)

##  Preprocessing Steps

- Encoded categorical features using `pandas.factorize()`
- Transformed `G3` into 3-class categorical target
- Applied Min-Max normalization
- Performed stratified train-validation-test split (60%-20%-20%)

## Machine Learning Models

| Model | Key Params | Accuracy |
|-------|------------|----------|
| **Decision Tree** | Entropy, depth=3 | **92%** |
| **SVM** | C=[0.1,1,10], kernel=linear/RBF | 87% |
| **KNN** | k=9, metric=Euclidean | 51% |

>  The **Decision Tree** model outperformed others with the best precision, recall, and interpretability.

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- Visualizations: Heatmaps, Tree plots, Histograms

## Results Summary

- **Decision Tree**: Best performer, interpretable, handled all classes well
- **SVM**: Strong on Fail/Excellent, weak on Pass class due to overlapping features
- **KNN**: Sensitive to noise and class imbalance, underperformed overall

