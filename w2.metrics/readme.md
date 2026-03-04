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
