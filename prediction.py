import csv
import pandas as pd
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
desease=['ARDS',
 'COVID-19',
 'COVID-19, ARDS',
 'Chlamydophila',
 'E.Coli',
 'Klebsiella',
 'Legionella',
 'No Finding',
 'Pneumocystis',
 'SARS',
 'Streptococcus']
data=[]
with open('drive/My Drive/metadata.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        temp=[]
        if((row[20])=='filename'):
          continue
        temp.append(row[20])
        if(row[4] in desease):
          temp.append(desease.index(row[4]))
        data.append(temp)
data=pd.DataFrame(data)
data1=[]
for i in range(0,len(data)):
  temp=[]
  if(".gz" in data[0][i]):
    continue
  temp.append(data[0][i])
  temp.append(data[1][i])
  data1.append(temp)
data1=pd.DataFrame(data1)
data=data1
inp=data[0]
out=data[1]
import tensorflow as tf
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same',input_shape=(500,500,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
inp=pd.DataFrame(data[0])
out=(data[1])
out2=[]
for i in out:
  temp=[]
  if(i==1 or i==2):
    
    temp.append(1)
    out2.append(temp)
  else:
    temp.append(0)
    out2.append(temp)
print(out2)
out=out2
out=pd.DataFrame(out)
c1=0
loca=[]
for i in out[0]:
  if(i==0):
    loca.append(c1)
  c1=c1+1
inptemp=[]
outtemp=[]
for j in range(4):
  for i in loca:
    temp=[]
    temp.append(inp[0][i])
    inptemp.append(temp)
    temp=[]
    temp.append(out[0][i])
    outtemp.append(temp)
inptemp=pd.DataFrame(inptemp)
outtemp=pd.DataFrame(outtemp)
inp=pd.concat([inp,inptemp],ignore_index=True)
out=pd.concat([out,outtemp],ignore_index=True)
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(inp,out, test_size = 0.5, random_state = 42)
xTrain=np.array(xTrain)
xTrain=pd.DataFrame(xTrain)
yTrain=np.array(yTrain)
yTrain=pd.DataFrame(yTrain)
xTest=np.array(xTest)
xTest=pd.DataFrame(xTest)
yTest=np.array(yTest)
yTest=pd.DataFrame(yTest)
yTrain
import matplotlib.pyplot as plt 
import matplotlib.image as img 
errori=[]
erroro=[]
inp1=[]
counter=0
precision=0
tp=0

re=0
tr=0

for i in (xTrain[0]):
  try:
     im1 = Image.open("drive/My Drive/"+i.strip())
     im1=im1.convert('P')
  except:
    continue
  newsize = (500,500) 
  im1 = im1.resize(newsize)
  a=list(im1.getdata())
  print(np.array(a).shape)
  try:
    a=(np.array(a).reshape(-1,500,500,1))
  except:
    errori.append(i)
    erroro.append(out[counter])
    counter=counter+1
    continue
  model.fit(a,np.matrix(yTrain[0][counter]))
  if(out[0][counter]==1):
    tp=tp+1
    if(model.predict_classes(a)[0][0]==1):
      precision=precision+1
  else:
    tr=tr+1
    if(model.predict_classes(a)[0][0]==0):
      re=re+1
  counter=counter+1
print((re+precision)/(tr+tp))
print(precision/tp,re/tr)
model.save('drive/My Drive/model.h5')
import tensorflow as tf
model1=tf.keras.models.Sequential()
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same',input_shape=(500,500,1)))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(tf.keras.layers.Conv2D(20,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.BatchNormalization())

model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(50, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.3))
model1.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model1.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
import matplotlib.pyplot as plt 
import matplotlib.image as img 
errori=[]
erroro=[]
inp1=[]
counter=0
precision=0
tp=0

re=0
tr=0

for i in (xTest[0]):
  try:
     im1 = Image.open("drive/My Drive/"+i.strip())
     im1=im1.convert('P')
  except:
    continue
  newsize = (500,500) 
  im1 = im1.resize(newsize)
  a=list(im1.getdata())
  print(np.array(a).shape)
  try:
    a=(np.array(a).reshape(-1,500,500,1))
  except:
    errori.append(i)
    erroro.append(out[counter])
    counter=counter+1
    continue
  model1.fit(a,np.matrix(yTest[0][counter]))
  if(out[0][counter]==1):
    tp=tp+1
    if(model1.predict_classes(a)[0][0]==1):
      precision=precision+1
  else:
    tr=tr+1
    if(model1.predict_classes(a)[0][0]==0):
      re=re+1
  counter=counter+1
print((re+precision)/(tr+tp))
print(precision/tp,re/tr)
model1.save('drive/My Drive/model1.h5')
import tensorflow as tf
model2=tf.keras.models.Sequential()
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same',input_shape=(500,500,1)))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.Conv2D(30,kernel_size=(3,3),activation='relu',padding='same'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.BatchNormalization())

model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(50, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.3))
model2.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model2.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
import matplotlib.pyplot as plt 
import matplotlib.image as img 
errori=[]
erroro=[]
inp1=[]
counter=0
precision=0
tp=0

re=0
tr=0

for i in (inp[0]):
  try:
     im1 = Image.open("drive/My Drive/"+i.strip())
     im1=im1.convert('P')
  except:
    continue
  newsize = (500,500) 
  im1 = im1.resize(newsize)
  a=list(im1.getdata())
  print(np.array(a).shape)
  try:
    a=(np.array(a).reshape(-1,500,500,1))
  except:
    errori.append(i)
    erroro.append(out[counter])
    counter=counter+1
    continue
  model2.fit(a,np.matrix(out[0][counter]))
  if(out[0][counter]==1):
    tp=tp+1
    if(model2.predict_classes(a)[0][0]==1):
      precision=precision+1
  else:
    tr=tr+1
    if(model2.predict_classes(a)[0][0]==0):
      re=re+1
  counter=counter+1
print((re+precision)/(tr+tp))
print(precision/tp,re/tr)
model2.save('drive/My Drive/model2.h5')
import tensorflow as tf
model3=tf.keras.models.Sequential()
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same',input_shape=(500,500,1)))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(tf.keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu',padding='same'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.BatchNormalization())

model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.Dense(50, activation='relu'))
model3.add(tf.keras.layers.Dropout(0.3))
model3.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model3.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])
import matplotlib.pyplot as plt 
import matplotlib.image as img 
errori=[]
erroro=[]
inp1=[]
counter=0
precision=0
tp=0

re=0
tr=0

for i in (inp[0]):
  try:
     im1 = Image.open("drive/My Drive/"+i.strip())
     im1=im1.convert('P')
  except:
    continue
  newsize = (500,500) 
  im1 = im1.resize(newsize)
  a=list(im1.getdata())
  print(np.array(a).shape)
  try:
    a=(np.array(a).reshape(-1,500,500,1))
  except:
    errori.append(i)
    erroro.append(out[counter])
    counter=counter+1
    continue
  model3.fit(a,np.matrix(out[0][counter]))
  if(out[0][counter]==1):
    tp=tp+1
    if(model2.predict_classes(a)[0][0]==1):
      precision=precision+1
  else:
    tr=tr+1
    if(model3.predict_classes(a)[0][0]==0):
      re=re+1
  counter=counter+1
print((re+precision)/(tr+tp))
print(precision/tp,re/tr)
model3.save('drive/My Drive/model3.h5')
from keras.models import load_model
import tensorflow as tf 
# load model
model = tf.keras.models.load_model('drive/My Drive/model.h5')
model1 = tf.keras.models.load_model('drive/My Drive/model1.h5')
model2 = tf.keras.models.load_model('drive/My Drive/model2.h5')

model3 = tf.keras.models.load_model('drive/My Drive/model3.h5')
l=len(xTest)
tans=0
tp=0
tn=0
cp=0
cn=0
for i in range(0,l):
  try:
    im1 = Image.open("drive/My Drive/"+xTest[0][i].strip())
    im1=im1.convert('P')
  except:
    continue
  newsize = (500,500) 
  im1 = im1.resize(newsize)
  a=list(im1.getdata())
  a=(np.array(a).reshape(-1,500,500,1))
  a1=model.predict_proba(a)
  a2=model1.predict_proba(a)
  a3=model2.predict_proba(a)
  #a4=model3.predict_proba(a)
  ans=(a1+a3+a2)/3
  ans=ans[0][0]
  ans=round(ans)
  if(yTest[0][i]==0):
    cn=cn+1
    if(ans==0):
      tn=tn+1
  if(yTest[0][i]==1):
    cp=cp+1
    if(ans==1):
      tp=tp+1
print(cn,cp,tp,tn)
print(tp/cp,tn/cn,(tp+tn)/(cp+cn))
