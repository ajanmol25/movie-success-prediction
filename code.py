import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



dataset=pd.read_csv(r"datasetfinal.csv")
dataset.head()
dataset.isnull().any()
dataset['actor_2_name'].fillna(dataset['actor_2_name'].mode()[0], inplace=True)
dataset['actor_3_name'].fillna(dataset['actor_3_name'].mode()[0], inplace=True)
x=dataset.iloc[:,[2,3,4,5,8]].values #Independent Variables
y=dataset.iloc[:,11].values #Dependent Variables
lb1 =LabelEncoder()
lb2=LabelEncoder()
lb3=LabelEncoder()
lb4=LabelEncoder()
x[:,0]=lb1.fit_transform(x[:,0])
x[:,1]=lb2.fit_transform(x[:,1])
x[:,2]=lb3.fit_transform(x[:,2])
x[:,3]=lb4.fit_transform(x[:,3])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Logistic Regression
logr=LogisticRegression()
logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)
a=accuracy_score(y_test,y_pred)
xnew=[["James Cameron","CCH Pounder","Joel David Moore","Wes Studi",237000000]]
xnew=np.array(xnew)
xnew[:,0]=lb1.fit_transform(xnew[:,0])
xnew[:,1]=lb1.fit_transform(xnew[:,1])
xnew[:,2]=lb1.fit_transform(xnew[:,2])
xnew[:,3]=lb1.fit_transform(xnew[:,3])
xnew=xnew.astype(object)
xnew_pred=logr.predict(xnew)

#SVM
'''model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred1=model.predict(x_test)
b=accuracy_score(y_test,y_pred1)
model.predict(xnew)'''

#Random Forest
rf=RandomForestClassifier(n_estimators=1000,criterion='entropy',random_state=0)
rf.fit(x_train,y_train)
y_pred2=rf.predict(x_test)
c=accuracy_score(y_test,y_pred2)
rf.predict(xnew)

#KNN
knn=KNeighborsClassifier(n_neighbors=5,p=2)
knn.fit(x_train,y_train)
y_pred3=knn.predict(x_test)
d=accuracy_score(y_test,y_pred3)
knn.predict(xnew)

#Decision Tree
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_pred4=dt.predict(x_test)
e=accuracy_score(y_test,y_pred4)
dt.predict(xnew)

#Kernel SVM
model2=SVC(kernel='rbf')
model2.fit(x_train,y_train)
y_pred5=model2.predict(x_test)
f=accuracy_score(y_test,y_pred5)
model2.predict(xnew)

#Predict New
i=input("Director Name : ")
j=input("Actor1 : ")
k=input("Actor2 : ")
l=input("Actor3: ")
m=int(input("Budget: "))
xnew=[[i,j,k,l,m]]
xnew=np.array(xnew)
xnew[:,0]=lb1.fit_transform(xnew[:,0])
xnew[:,1]=lb2.fit_transform(xnew[:,1])
xnew[:,2]=lb3.fit_transform(xnew[:,2])
xnew[:,3]=lb4.fit_transform(xnew[:,3])
xnew=xnew.astype(object)
final=dt.predict(xnew)
print(final[0])
