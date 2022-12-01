import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from PIL import Image

st.set_page_config(
    page_title="Linear regression",
    page_icon="🦄",
)
st.title('Linear regression')
with st.expander("Giới thiệu", True):
    st.write("""Linear regression (Hồi quy tuyến tính)là một thuật toán supervised, ở đó quan hệ giữa đầu vào và đầu ra được mô tả bởi một hàm tuyến tính. Thuật toán này còn được gọi là linear fitting hoặc linear least square""")
    image = Image.open('pages/HoiQuy.png')
    st.image(image, caption="""Phân bố của các điểm dữ liệu (màu đỏ) và đường thẳng xấp xỉ tìm được bởi linear regression.""")
   
def HoiQuy_Bai01():
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
    one = np.ones((1, X.shape[1]))
    Xbar = np.concatenate((one, X), axis = 0) # each point is one row
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    A = np.dot(Xbar, Xbar.T)
    b = np.dot(Xbar, y)
    w = np.dot(np.linalg.pinv(A), b)
    # weights
    w_0, w_1 = w[0], w[1]
    print(w_0, w_1)
    y1 = w_1*155 + w_0
    y2 = w_1*160 + w_0
    st.write('Input 155cm, true output 52kg, predicted output %.2fkg' %(y1))
    st.write('Input 160cm, true output 56kg, predicted output %.2fkg' %(y2))

def HoiQuy_Bai02():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("scikit-learn' solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
    fig = plt.figure() 
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]

    plt.plot(x, y)
    st.pyplot(fig) 

def HoiQuy_Bai03():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    X2 = X**2
    X_poly = np.hstack((X, X2))

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, y)
    st.write("lin_reg.intercept_= ", lin_reg.intercept_)
    st.write("lin_reg.coef_= ",lin_reg.coef_)
    a = lin_reg.intercept_[0]
    b = lin_reg.coef_[0,0]
    c = lin_reg.coef_[0,1]
    st.write("a = ",a)
    st.write("b = ", b)
    st.write("c = ", c)

    x_ve = np.linspace(-3,3,m)
    y_ve = a + b*x_ve + c*x_ve**2
    fig = plt.figure()  
    plt.plot(X, y, 'o')
    plt.plot(x_ve, y_ve, 'r')

    loss = 0 
    for i in range(0, m):
        y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
        sai_so = (y[i] - y_mu)**2 
        loss = loss + sai_so
    loss = loss/(2*m)
    st.write('loss = %.6f' % loss)

    y_train_predict = lin_reg.predict(X_poly)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số bình phương trung bình: %.6f' % (sai_so_binh_phuong_trung_binh/2))
    st.pyplot(fig) 

def HoiQuy_Bai04():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("scikit-learn's solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
    fig = plt.figure()
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]


    plt.plot(x, y)
    st.pyplot(fig) 
def HoiQuy_Bai05():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    huber_reg = linear_model.HuberRegressor()
    huber_reg.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("scikit-learn's solution : w_1 = ", huber_reg.coef_[0], "w_0 = ", huber_reg.intercept_)
    fig = plt.figure()
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = huber_reg.coef_[0]
    b = huber_reg.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.pyplot(fig) 

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🐎Bài 01", "🦓Bài 02", "🐴Bài 03", "🌈Bài 04","💈Bài 05"])

with tab1:
    st.header("🐎Bài 01")
    HoiQuy_Bai01()

with tab2:
    st.header("🦓Bài 02")
    HoiQuy_Bai02()
with tab3:
   st.header("🐴Bài 03")
   HoiQuy_Bai03()

with tab4:
    st.header("🌈Bài 04")
    HoiQuy_Bai04()
with tab5:
    st.header("💈Bài 05")
    HoiQuy_Bai05()
