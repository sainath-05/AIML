import pandas as pd
path="C:\\Users\\bond\\Desktop\\batch4\\21_dementiadiseasetypeprediction\\dementia_dataset.csv"
data=pd.read_csv(path)
print(data)
print(data.info())
#print(data.isnull().sum())
medianvalue=data.SES.median()
#print(medianvalue)
data.SES=data.SES.fillna(medianvalue)
#print(data)
medianvalue=data.MMSE.median()
#print(medianvalue)
data.MMSE=data.MMSE.fillna(medianvalue)
#print(data)
#print(data.isnull().sum())
data['Group']=data['Group'].map({'Nondemented':1,'Demented':2,'Converted':3})
data['M/F']=data['M/F'].map({'M':1,'F':0})
data['Hand']=data['Hand'].map({'R':1})
import sklearn
from sklearn.preprocessing import LabelEncoder
le_MF = LabelEncoder()
data['Gender'] = le_MF.fit_transform(data['M/F'])
inputs= data.drop(['Group', 'Subject ID', 'MRI ID','M/F'], axis=1)
output = data['Group']
print(inputs)
print(output)
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
import numpy as np
new_inputs = np.array([[Visit, MR_Delay, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF, Gender]])
new_inputs_scaled = sc.transform(new_inputs)
prediction = model.predict(new_inputs_scaled)
print("Predicted Group:", prediction)