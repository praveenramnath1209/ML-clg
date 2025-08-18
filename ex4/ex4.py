#import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


#import the dataset
data = pd.read_csv('ex4.csv')
data.head()

#create dataframe
df = pd.DataFrame(data)

#features and target
x = df[['study_hr','attendence']]
y = df[['result']]

model = DecisionTreeClassifier(criterion='entropy',random_state=0)
model.fit(x,y)

#visualize the decision tree
plt.figure(figsize=(8,8))
plot_tree(model,feature_names=['study_hr','attendance'],class_names=['1','0'],filled=True)

plt.show()

#example prediction
new = [[5, 85]]
prediction = model.predict(new)

print("Prediction for new student:","1" if prediction[0] ==  1 else "0")
