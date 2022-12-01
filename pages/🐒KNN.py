import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from skimage import exposure
import imutils
import cv2
import joblib
from tensorflow import keras 
from PIL import Image

st.set_page_config(
    page_title="K-nearest neighbor",
    page_icon="🐒",
)

st.title('K-nearest neighbor')
with st.expander("Giới thiệu", True):
    st.write("""K-nearest neighbor (KNN) là một trong những thuật toán supervised learning đơn giản. 
    Khi huấn luyện, thuật toán này không học một điều gì từ dữ liệu huấn luyện mà nhớ lại một cách máy móc toàn bộ dữ liệu đó. 
    Đây cũng là lý do thuật toán này được xếp vào loại lazy learning, mọi tính toán được thực hiện khi nó cần dự đoán đầu ra của dữ liệu mới. 
    KNN có thể áp dụng được vào cả classification và regression. KNN còn được gọi là một thuật toán instance-based hay memory-based learning.""")
    image = Image.open('pages/KNN.png')
    st.image(image, caption="""Ví dụ về 1NN. Các hình tròn là các điểm dữ liệu huấn luyện. Các hình khác màu thể hiện các lớp khác nhau. Các
                                vùng nền thể hiện các điểm được phân loại vào lớp có màu tương ứng khi sửdụng 1NN""")

def KNN_bai01():
    np.random.seed(100)
    N = 150
    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)
    nhom_0 = []
    nhom_1 = []
    nhom_2 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])
        else:
            nhom_2.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    nhom_2 = np.array(nhom_2)
    fig = plt.figure() 
    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)
    plt.plot(nhom_2[:,0], nhom_2[:,1], 'ob', markersize = 2)
    plt.legend(['Nhóm 0', 'Nhóm 1', 'Nhóm 2'])
    st.pyplot(fig)    
    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 
    # default k = n_neighbors = 5
    #         k = 3
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write('Sai số: ', sai_so)
    
    my_test = np.array([[2.5, 4.0]])
    ket_qua = knn.predict(my_test)
    st.write('Kết quả nhận dạng là nhóm: ', ket_qua[0])

def KNN_bai02():
    mnist = datasets.load_digits()
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
    mnist.target, test_size=0.25, random_state=42)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
    test_size=0.1, random_state=84)

    st.write("Training data points: ", len(trainLabels))
    st.write("Validation data points: ", len(valLabels))
    st.write("Testing data points: ", len(testLabels))
    
    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)
    score = model.score(valData, valLabels)
    st.write("Accuracy = %.2f%%" % (score * 100))
    st.markdown("""___""")
    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
        
        image = testData[i]
        prediction = model.predict(image.reshape(1, -1))[0]
        image = image.reshape((8, 8)).astype("uint8")

        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

        st.write("Tôi nghĩ đó là số {}".format(prediction))
        st.image(image, width = 100, clamp=True) 

def KNN_bai03():

    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

    RESHAPED = 784
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED) 
    (trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train,
        test_size=0.1, random_state=84)

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)
    joblib.dump(model, "pages/knn_mnist.pkl")

    predicted = model.predict(valData)
    do_chinh_xac = accuracy_score(valLabels, predicted)
    st.write('Độ chính xác trên tập validation: %.0f%%' % (do_chinh_xac*100))
 
    # Đánh giá trên tập test
    predicted = model.predict(X_test)
    do_chinh_xac = accuracy_score(Y_test, predicted)
    st.write('Độ chính xác trên tập test: %.0f%%' % (do_chinh_xac*100))

def KNN_bai03a():
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]


    # 784 = 28x28
    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    knn = joblib.load("pages/knn_mnist.pkl")
    predicted = knn.predict(sample)
    k = 0
    arr =  np.zeros([10, 10], dtype=int)
    for x in range(0, 10):
        for y in range(0, 10):
            arr[x][y] = predicted[k]
            k = k + 1
    st.write(arr)    
        
  
    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1

    st.image(digit, width=400)

def KNN_bai04():
    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    X_test = X_test
    knn = joblib.load("pages/knn_mnist.pkl")
    if st.button("Tạo chữ số và nhận dang"):
        index = np.random.randint(0, 9999, 100)
        digit = np.zeros((10*28,10*28), np.uint8)
        k = 0
        for x in range(0, 10):
            for y in range(0, 10):
                digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
                k = k + 1
        cv2.imwrite('pages/digit.jpg', digit)
        st.image('pages/digit.jpg')

        #Recognition
        sample = np.zeros((100,28,28), np.uint8)
        for i in range(0, 100):
            sample[i] = X_test[index[i]]

        RESHAPED = 784
        sample = sample.reshape(100, RESHAPED) 
        predicted = knn.predict(sample)
        arr =  np.zeros([10, 10], dtype=int)
        k = 0
        for x in range(0, 10):
            for y in range(0, 10):
                arr[x][y] = predicted[k]
                k = k + 1
        st.write(arr)  

def KNN_bai08():
    st.image('pages/castle.jpg')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🙈Bài 01", "🙉Bài 02", "🙊Bài 03", "🐵Bài 03a", "🍌Bài 04", "🌴Bài 08"])

with tab1:
   st.header("🙈Bài 01")
   KNN_bai01()

with tab2:
   st.header("🙉Bài 02")
   KNN_bai02()

with tab3:
   st.header("🙊Bài 03")
   KNN_bai03()

with tab4:
   st.header("🐵Bài 03a")
   KNN_bai03a()

with tab5:
   st.header("🍌Bài 04")
   KNN_bai04()

with tab6:
   st.header("🌴Bài 08")
   KNN_bai08()
