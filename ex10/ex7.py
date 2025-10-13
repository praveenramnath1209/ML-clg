 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("dataset.csv")
x = data.drop("Outcome", axis=1)

y = data["Outcome"]
xtr, xte, ytr, yte = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

base = DecisionTreeClassifier(max_depth=1)  # Weak learner (stump)  a
ada = AdaBoostClassifier(estimator=base, n_estimators=100, random_state=42)
ada.fit(xtr, ytr) 
ypr = ada.predict(xte) 

print(f"Accuracy: {accuracy_score(yte, ypr):.2f}")   
print("\nClassification Report:\n", classification_report(yte,ypr))
cm = confusion_matrix(ype ,  ytr)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - AdaBoost")
plt.show()
