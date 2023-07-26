# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df=pd.read_csv('diabetes.csv')
zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
for column in zero_not_accepted:
    df[column] = df[column].replace(0,np.NaN)
    mn = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN,mn)
df1=df.dropna(inplace=True)
print(df)


# %%
from sklearn.model_selection import train_test_split as spl
X=df.iloc[:,0:8]
y=df.iloc[:,8]
print(y)
X_train,X_test,y_train,y_test=spl(X,y,test_size=0.2)

# %%
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True)

# %%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns)
X_test=pd.DataFrame(scaler.transform(X_test),columns=X.columns)
print(X_test)

# %%
print(type(X_train))
print(type(y_train))
print(type(X_test))
print(type(y_test))
print(y_test.shape)

# %%
#Random Sampling as Validation data
df2=df.sample(80)
x_val=df2.iloc[:,:-1]
x_val=pd.DataFrame(scaler.transform(x_val),columns=x_val.columns)
y_val=df2.iloc[:,-1]
print(y_val)
print(x_val)



# %%
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Dropout
#defining the ann model
model=Sequential()
model.add(Dense(32,input_dim=8,kernel_initializer='normal',kernel_regularizer=regularizers.l2(.001),activation='relu'))
#adding neural layers
model.add(Dropout(0.25))
#diconnecting non essential connections of neurons
model.add(Dense(32,kernel_initializer='normal',kernel_regularizer=regularizers.l2(.001),activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16,kernel_initializer='normal',kernel_regularizer=regularizers.l2(.001),activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8,kernel_initializer='normal',kernel_regularizer=regularizers.l2(.001),activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# %%
#fitting the data
history=model.fit(X_train,y_train,validation_data=(x_val,y_val),epochs=350,batch_size=8)

# %%
#predicting outcomes
y_pred=(model.predict(x_val)>.5).astype('int32')
y_pred.shape

# %%
from sklearn.metrics import confusion_matrix,precision_score,matthews_corrcoef
from sklearn.metrics import f1_score
cm1=confusion_matrix(y_val,y_pred)
print('Confusion Matrix :\n',cm1)
total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
sensitivity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
specificity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
f1_score=f1_score(y_val,y_pred)
prece=precision_score(y_val,y_pred)
print("Accuracy : " + str(accuracy1) )
print("Sensitivity : "+ str(sensitivity1))
print("Specificity : "+ str(specificity1))
print("F1_Score : "+ str(f1_score))
print("Precetion : " + str(prece))

# %%
train_loss=history.history['loss']
test_loss=history.history['val_loss']
epoch_ct=range(1,len(train_loss)+1)
plt.plot(epoch_ct,train_loss,'r--')
plt.plot(epoch_ct,test_loss,'b-')
plt.legend(['Train Loss','Validation'])
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.show()

# %%
from sklearn.metrics import roc_curve
y_pred_keras=model.predict(X_train).ravel()
fpr,tpr,thresholds=roc_curve(y_train,y_pred_keras)
plt.plot([0,1],[0,1])
plt.plot(fpr,tpr)
plt.xlabel('True Possitive Rate')
plt.ylabel('False Possitive Rate')
plt.show()

#%% Prediction Base
input_array = (5,166,72,19,175,25.8,0.587,51)

#Converting to numpy_array

array = np.asarray(input_array)

#Scaling Input

input_scaled = array.reshape(1,-1)

#predicting

prediction = model.predict(input_scaled)
print(prediction)

if prediction[0] == 0 :
    print("Person Does Not Have Diabetes")
else:
    print("Person Has Diabetes")
# %% Saving the Trained Model
import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
# %% Loading Saved Model
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_array = [1,103,30,38,83,43.3,0.183,33]

#Converting to numpy_array

array = np.asarray(input_array)

#Scaling & Standerizing Input

input_scaled = array.reshape(1,-1)
std_data = scaler.fit_transform(input_scaled)

#predicting

prediction = loaded_model.predict(std_data)
print(prediction)

if prediction[0] <= 0.5 :
    print("Person Does Not Have Diabetes")
else:
    print("Person Has Diabetes")

# %%
