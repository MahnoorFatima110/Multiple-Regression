
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score

dataset=pd.read_csv("dataset.csv")

X=dataset.iloc[:,[0,1]].values
Y=dataset.iloc[:,2:3].values

X_train,X_test,Y_train,Y_test=tts(X,Y,train_size=0.6,random_state=1)



model=LR()
model.fit(X_train,Y_train)

res=model.predict(X_test)
