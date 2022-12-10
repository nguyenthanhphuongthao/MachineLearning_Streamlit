import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.svm import SVC
import streamlit as st

st.title("Ứng dụng SVM vào việc phân loại hình MRI phát hiện khối u tuyến yên")
# Prepare/collect data
path = os.listdir('pages/data/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}

X = []
Y = []
for cls in classes:
    pth = 'pages/data/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])

X = np.array(X)
Y = np.array(Y)

#Prepare data
X_updated = X.reshape(len(X), -1)


#Split Data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)
st.write("Kích thước của xtrain và xtest sau khi chia nhỏ dữ liệu:")
xtrain.shape, xtest.shape

#Feature Scaling
st.write ("Giá trị lớn nhất và nhỏ nhất của xtrain sau khi chia nhỏ dữ liệu")
st.write(xtrain.max(), xtrain.min())
st.write ("Giá trị lớn nhất và nhỏ nhất của xtest sau khi chia nhỏ dữ liệu")
st.write(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
st.write('Sau khi lần lượt chia các giá trị trong xtrain cho 225 ta được giá trị lớn nhất và nhỏ nhất là' ,xtrain.max() , xtrain.min())
st.write('Sau khi lần lượt chia các giá trị trong xtest cho 225 ta được giá trị lớn nhất và nhỏ nhất là', xtest.max() ,xtest.min())

#Train Model
sv = SVC()
sv.fit(xtrain, ytrain)

#Evaluation
## svm
st.write("Đáng giá kết quả:")
st.write("Training Score:", sv.score(xtrain, ytrain))
st.write("Testing Score:", sv.score(xtest, ytest))

#Prediction
pred = sv.predict(xtest)

#TEST MODEL
dec = {0:'No Tumor', 1:'Positive Tumor'}
st.write("Kết quả dự đoán của model trên tập dữ liệu hình MRI không có khối u tuyến yên:")
fig1 = plt.figure(figsize=(12,8))
p = os.listdir('pages/data/Testing/')
c=1
for i in os.listdir('pages/data/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('pages/data/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    c+=1
st.pyplot(fig1)

st.write("Kết quả dự đoán của model trên tập dữ liệu hình MRI có khối u não:")
fig2 = plt.figure(figsize=(12,8))
p = os.listdir('pages/data/Testing/')
c=1
for i in os.listdir('pages/data/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('pages/data/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    c+=1
st.pyplot(fig2)
