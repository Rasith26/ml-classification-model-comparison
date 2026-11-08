Online Shoppers Purchasing Intention Prediction

1. Problem Statement:
The objective of this project is to predict whether an online visitor will make a purchase during a browsing session using machine learning classification algorithms. The project implements and compares five classification models using the Online Shoppers Purchasing Intention dataset.

2. Dataset Description:
The Online Shoppers Purchasing Intention dataset is obtained from the UCI Machine Learning Repository.
- Dataset name: Online Shoppers Purchasing Intention
- Dataset source: UCI Machine Learning Repository
- Dataset type: Classification
- Target variable: Revenue
- Target values: True and False
- Number of records: 12,330
- Number of input features: 17
The dataset contains numerical and categorical information related to online shopping sessions.

3. GitHub Repository Link
Link: https://github.com/Rasith26/ml-classification-model-comparison

4. Models Used
The following five classification algorithms are implemented:

i. Logistic Regression: Logistic Regression is a supervised classification algorithm that uses the logistic function to estimate the probability of a binary outcome.

ii. Decision Tree Classifier: Decision Tree is a supervised learning algorithm that uses decision rules to divide data into different classes.

iii. K-Nearest Neighbors: KNN classifies a new observation based on the majority class of its nearest neighboring observations.

iv. Gaussian Naive Bayes: Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem and the assumption of conditional independence between features.

v. Support Vector Machine: SVM is a supervised learning algorithm that identifies an optimal decision boundary to separate different classes.

5. Evaluation Metrics
The models are evaluated using:
- Accuracy
- AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

6. Model Comparison
MODEL COMPARISON
               Model  Accuracy      AUC  Precision   Recall  F1-score      MCC
 Logistic Regression  0.849959 0.896200   0.510714 0.748691  0.607219 0.533041
       Decision Tree  0.853204 0.704697   0.528249 0.489529  0.508152 0.422449
                 KNN  0.874696 0.799768   0.671362 0.374346  0.480672 0.438888
Gaussian Naive Bayes  0.673560 0.793161   0.293659 0.787958  0.427861 0.323431
                 SVM  0.856042 0.885570   0.525813 0.719895  0.607735 0.531852

7. Streamlit Community Cloud Link
Link: https://ml-classification-model-comparison-qcnesdauysfirclrtymm4k.streamlit.app/