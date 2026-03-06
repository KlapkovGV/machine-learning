## Confusion Matrix in Machine Learning

A **Confusion Matrix** is a performance measurement for machine learning classification problems. It is a table that allows visualization of the performance of an algorithm by comparing predicted values against actual values. 

In Binary Classification (where there are only two possible outcomes), the performance is evaluated using these four outcomes: TP, TN, FP, and FN.

Let's take an example: AI-Driven "System Health Check" for edge devices. We have an AI model monitoring a robotic arm on an assembly line. The goal is to predict if the robotic arm will experience a critical hardware failure in the next 24 hours.

- **Positive Class (+)**: system is Failing (needs maintenance);
- **Negative Class (-)**: system is Normal (keep running).

**True Positive (TP) - The Successful Catch**

The AI predicts the robot is failing, and in reality, a bearing was about to snap.

**True Negative (TN) - The Smooth Operation**

The AI predicts the system is normal, and the robot continues to work perfectly.

**False Positive (FP) - The False Alarm**

The AI predicts a failure, but when the engineers check the robot, it is actually in normal condition.

**False Negative (FN) - The Expensive Miss**

The AI predicts the system is normal, so there is nothing to do. However, the robot breaks down two hours later. This is the most dangerous error in this context. 

### The matrix structure

| | Actual: Positive | Actual: Negative |
| :--- | :--- | :--- |
| **Predicted: Positive** | **TP** (True Positive) | **FP** (Type I Error) |
| **Predicted: Negative** | **FN** (Type II Error) | **TN** (True Negative) |

## Performance / Evaluation Metrics: Accuracy

**Accuracy** is the most intuitive performance measure. It is simply the ratio of correctly predicted observations to the total observations.

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Total Correct Predictions}}{\text{Total Number of Samples}}$$

It works best for **Balanced Datasets**, where the number of samples in each class (Positive and Negative) is roughly equal. Example: a dataset with 500 spam emails and 500 non-spam emails. The algorithm correctly finds patterns and classifies the dataset. 

The accuracy trap 

Accuracy alone can be very misleading when dealing with **imbalanced datasets** where one class is much more frequent than the other. 

Imagine we are monitoring 1,000 robots at a plant. 990 robots are in normal condition, 10 robots are failing. 

If we build a model that simply predicts normal for every single robot, we will get the results like TN = 990, TP = 0. Accuracy will be 99%.

Even though the accuracy is 99%, the model is useless. It failed to catch the 10 failures that would stop the production line. In this case, accuracy makes the model appear successful while it is actually failing its primary goal. 

We can use the confusion matrix for binary examples. But what if we build 100 models? Or if multi-class classification is used? To truly evaluate a model's performance on imbalanced data, we must look beyond Accuracy and use metrics like Precision, Recall, and the F1-Score. 

## Performance / Evaluation Metrics: Precision

While Accuracy looks at the whole picture, Precision focuses on the quality of our positive predictions. 

Precision measures the proportion of correctly predicted positive samples out of all samples that the model predicted as positive. 

$$Precision = \frac{TP}{TP + FP} = \frac{\text{Correctly Predicted Positives}}{\text{Total Predicted Positives}}$$

Precision is a useful metric in cases where **False Positives (FP)** are a higher concern than False Negatives. It is about **quality and reliability**. High precision means that when the model says Positive (it is broken). 

In the context of the platform for system health checks, Precision is critical for managing maintenance costs. The AI monitors a system and predicts a critical failure (Positive). The cost of a false positive when the model triggers a fire alarm. An engineer is sent to the edge device, and production is potentially paused for a check, only to find the system was actually healthy. This wastes specialized labor hours and decreases operational efficiency. 

## Performance / Evaluation Metrics: Recall

Recall is a measure of actual observations which are predicted correctly. It shows how many observations of the positive class were actually identified as positive. 

$$Recall = \frac{TP}{TP + FN} = \frac{\text{Correctly Predicted Positives}}{\text{Total Actual Positives}}$$

It is also known as Sensitivity. Recall is a valid choice of evaluation metric when the goal is to capture as **many positives as possible**. Recall is important when minimizing False Negatives is more critical than minimizing False Positives.

For example, missing a critical hardware failure (False Negative) results in missed opportunities or catastrophic equipment damage. 

## Performance / Evaluation Metrics: F1-Score

The F1-score is the harmonic mean of precision and recall, and provides a balanced measure that combines both metrics. It is a good choice when both precision and recall are equally important, and it helps in finding a trade-off between precision and recall. F1-score is often used when there is an uneven class distribution or **class imbalance**, and it provides a single value that summarizes the model's performance.

If our precision is low, the F1 is low and if the recall is low, again our F1 score is low. A higher F1-score indicates a better balance between precision and recall. A perfect F1-score of 1.0 is rarely achievable in real-world scenarios. 

$$F1 score = \frac{2 * TP}{2 * TP + FP + FN} = 2 * \frac{Recall * Precision}{Recall + Precision}$$

__
Use:
- **accuracy** for balanced datasets;
- **precision** when minimizing False Positives is critical;
- **recall** when minimizing False Negatives is crucial;
- **F1-score** when both False Positives and False Negatives are equally important, especially in imbalanced datasets;

The ideal value for Accuracy, Precision, Recall, and F1-score would be 1.0 or near to 1.

Once the model is trained, it is crucial to understand how well it performs. We can do this by generating a confusion matrix.

```python
import pandas as pd, numpy as np, matplotlob.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

y_pred = df["predicted_label"]
y_true = df["true_label"]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```

Here are the evaluation metrics.
```python
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, average='weighted')

revall = recall_score(y_true, y_pred, average='weighred')

f1 = f1_score(y_true, y_pred, average='weighted')

report = classification_report(y_true, y_pred)
```

For multi-class classification, evaluating the model across all categories is essential. 

```python
cm = confusion_matrix(y_true, y_pred)

class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)

ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(class_labels, rotation=0, ha='right', fontsize=10)

ax.set_title('Confusion Matrix', fontsize=14)

plt.tight_layout()
plt.show()
```

## Undersatnding the ROC-AUC Curve

ROC stands for Receiver Operating Characteristic. The ROC curve visualizes the performance of a binary classifier across all possible thresholds. The curve tells us how well the model can distinguish between two classes (e.g., Disease vs. No Disease). A better model can accurately distinguish between the two, whereas a poor model will have difficulty distinguishing between them.

Let's assune we have a model which predicts whether the patient has a particular disease or no. Here, the red distribution represents all the patients who do not have the disease and the green distribution represents all the patients who have the disease. 

<img width="320" height="216" alt="image" src="https://github.com/user-attachments/assets/8f9f1d62-d82e-449f-8dff-51dd093a4f03" />

Now we got to pick a value where we need to set the cut off i.e. a threshold value, abouve which we will predict everyone as positive and below which will predict as negative. We will set the threshold at 0.5.

<img width="320" height="214" alt="image" src="https://github.com/user-attachments/assets/9961e9bb-679f-4467-9e50-d7d252bff8de" />

This images illustrates how a machine learning model makes decisions and how we measure its accuracy. The graph on the right shows an S-curve, which is the result of a logistic regression model. This curve takes our data and turns it into a probability between 0 and 1. The blue points on the curve represnt specific predictions, such as a 30% or 80% probability that an event will happen. 

To turn these probabilities into a final yes or no answer, the model uses a threshold value, which is usually set at 0.5. Any point above this threshold is classified as a positive result, while any point below it is classified as a negative result. Once these classification are made, we compare them to the actual truth to create a confusion matrix. 

The confusion matrix is a table that tracks four specific outcomes: TP and TN, which represent correct predictions, and FP and FN, which represent errors. Understanding these four elements is the first step toward building an ROC-AUC curve, because that curve is created by calculating how these error rates changes as you move the threshold line higher or lower. Together, these tools allow us to evaluate the efficiency and precision of classification model. 
