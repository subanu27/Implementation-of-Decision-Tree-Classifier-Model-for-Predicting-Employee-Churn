# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy score of the model.
6. Check the trained model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Subanu. K
RegisterNumber:  212219040152
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project",
"average_montly_hours","time_spend_company","Work_accident",
"promotion_last_5years","salary"]]
x.head()
y=df["left"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
acc
dt.predict([[.5,.8,9,260,6,0,1,2]])
```

## Output:
Initial dataset:

![image1](https://user-images.githubusercontent.com/87663343/173234345-825dd077-46ee-46a7-b4a1-da5829eb11dc.png)


Dataset Information:

![img2](https://user-images.githubusercontent.com/87663343/173234397-8e4f8bd6-7eac-4cd6-a762-3000603ed5a7.png)


Null dataset:

![img3](https://user-images.githubusercontent.com/87663343/173234434-b16019d3-71cb-498e-9c0e-5713440e8c24.png)


Values counts in left column:
![img4](https://user-images.githubusercontent.com/87663343/173234493-a60f9dbe-8aa5-4a13-8cf6-924e35b66da9.png)


Encoded dataset:

![img5](https://user-images.githubusercontent.com/87663343/173234527-7a810f2e-9859-4da1-ad40-6af633c9ebd7.png)


X set:

![img6](https://user-images.githubusercontent.com/87663343/173234554-310937f7-6252-423b-96e0-fcf88840e85b.png)

Y values:

![img7](https://user-images.githubusercontent.com/87663343/173234580-8f749de5-cfd2-499d-bac7-d2dea58ca7ec.png)


Accuracy score:

![img8](https://user-images.githubusercontent.com/87663343/173234599-039d9846-0280-4628-bf62-44f7014c7123.png)


Dataset Prediction:

![img9](https://user-images.githubusercontent.com/87663343/173234635-8a302428-a93e-4121-9382-ea455dd65237.png)











## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
