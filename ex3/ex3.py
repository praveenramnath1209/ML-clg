import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

data=pd.read_csv("ex3.csv")
data.head()

data['Gender']=LabelEncoder().fit_transform(data['Gender'])

x=data[['Age','Gender','bmi','bp','cholesterol']]
y=data['Condition']
scaler=StandardScaler()
xscale=scaler.fit_transform(x)
xtr,xte,ytr,yte=train_test_split(xscale,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(xtr,ytr)
predict=model.predict(xte)
prob=model.predict_proba(xte)[:,1]
print("Accuracy: ",accuracy_score(yte,predict))
print("Classification Report \n",classification_report(yte,predict,zero_division=1))
cm=confusion_matrix(yte,predict)

disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confussion Matrix")
plt.show()

new=pd.DataFrame([[60,1,27,130,200]],colums=["Age","Gender","BMI","BP","Cholestrol"])
newscale=scaler.transform(new)
newcondition=model.predict_proba(newscale)[0][1]
print(f"Probability of Devleoping the condition: {newcondition:.2f}")