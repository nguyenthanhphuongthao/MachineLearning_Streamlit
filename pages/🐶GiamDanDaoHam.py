from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Giảm dần đạo hàm",
    page_icon="🐶",
)

st.title('Giảm dần đạo hàm')

X = np.random.rand(1000)
y = 4 + 3 * X + .5*np.random.randn(1000)
model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
w, b = model.coef_[0][0], model.intercept_[0]
# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

def intro():
    st.title('Giảm dần đạo hàm')
    st.sidebar.success("Select a demo above.")

def grad1(x):
    return 2*x+ 5*np.cos(x)
def cost1(x):
    return x**2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad1(x[-1])
        if abs(grad1(x_new)) < 1e-3: # just a small number
            break
        x.append(x_new)
    return (x, it)

def GiamDanDaoHam_Bai01():
    x0 = -5
    eta = 0.1
    (x, it) = myGD1(x0, eta)
    x = np.array(x)
    y = cost1(x)

    n = 101
    xx = np.linspace(-6, 6, n)
    yy = xx**2 + 5*np.sin(xx)
    fig = plt.figure()
    plt.subplot(2,4,1)
    plt.plot(xx, yy)
    index = 0
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,2)
    plt.plot(xx, yy)
    index = 1
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,3)
    plt.plot(xx, yy)
    index = 2
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,4)
    plt.plot(xx, yy)
    index = 3
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,5)
    plt.plot(xx, yy)
    index = 4
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,6)
    plt.plot(xx, yy)
    index = 5
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,7)
    plt.plot(xx, yy)
    index = 7
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,8)
    plt.plot(xx, yy)
    index = 11
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad1(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.tight_layout()
    st.pyplot(fig) 

def GiamDanDaoHam_Bai02():
    x0 = 0
    x1 = 1
    y0 = w*x0 + b
    y1 = w*x1 + b
    fig = plt.figure()
    plt.plot(X, y, 'bo', markersize = 2)
    plt.plot([x0, x1], [y0, y1], 'r')
    st.pyplot(fig) 

def grad2(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost2(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

def myGD2(w_init, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad2(w[-1])
        if np.linalg.norm(grad2(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it)
def GiamDanDaoHam_Bai02a():
    sol_sklearn = np.array([b, w])
    st.write('Solution found by sklearn:', sol_sklearn)
    w_init = np.array([0, 0])
    (w1, it1) = myGD2(w_init, 1)
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))


np.random.seed(100)
N = 1000


def grad3(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost3(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

def myGD3(w_init, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad3(w[-1])
        if np.linalg.norm(grad3(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it)
def GiamDanDaoHam_Bai03():
    w, b = model.coef_[0][0], model.intercept_[0]
    st.write('b = %.4f va w = %.4f' % (b, w))
    w_init = np.array([0, 0])
    (w1, it1) = myGD3(w_init, 1)
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    for item in w1:
        st.write(item, cost3(item))

    st.write(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F
    fig = plt.figure() 
    temp = w1[0]
    bb = temp[0]
    ww = temp[1]
    zz = cost3(temp) 
    ax = plt.axes(projection="3d")
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[1]
    bb = temp[0]
    ww = temp[1]
    zz = cost3(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[2]
    bb = temp[0]
    ww = temp[1]
    zz = cost3(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[3]
    bb = temp[0]
    ww = temp[1]
    zz = cost3(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)


    ax.plot_wireframe(b, w, z)
    ax.set_xlabel("b")
    ax.set_ylabel("w")
    st.pyplot(fig) 

def GiamDanDaoHam_Bai04():
    x = np.linspace(-2, 2, 21)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    fig = plt.figure() 
    plt.contour(X, Y, Z, 10)
    st.pyplot(fig)

def grad4(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost4(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

def myGD4(w_init, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad4(w[-1])
        if np.linalg.norm(grad4(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it)
def GiamDanDaoHam_Bai05():
    w, b = model.coef_[0][0], model.intercept_[0]
    st.write('b = %.4f va w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)
    w_init = np.array([0, 0])
    (w1, it1) = myGD4(w_init, 1)
    st.write('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    for item in w1:
        st.write(item, cost4(item))

    st.write(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F
    fig = plt.figure() 
    plt.contour(b, w, z, 45)
    bdata = []
    wdata = []
    for item in w1:
        plt.plot(item[0], item[1], 'ro', markersize = 3)
        bdata.append(item[0])
        wdata.append(item[1])

    plt.plot(bdata, wdata, color = 'b')

    plt.xlabel('b')
    plt.ylabel('w')
    plt.axis('square')
    st.pyplot(fig) 

def GiamDanDaoHam_Temp():
    fig = plt.figure() 
    ax = plt.axes(projection="3d")

    X = np.linspace(-2, 2, 21)
    Y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(X, Y)
    Z = X*X + Y*Y
    ax.plot_wireframe(X, Y, Z)
    st.pyplot(fig) 

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🐕Bài 01", "🐩Bài 02", "🐕‍🦺Bài 02a", "🐈Bài 03", "🐈‍⬛Bài 04", "🦴Bài 05", "🐠Temp"])

with tab1:
   st.header("🐕Bài 01")
   GiamDanDaoHam_Bai01()

with tab2:
   st.header("🐩Bài 02")
   GiamDanDaoHam_Bai02()

with tab3:
   st.header("🐕‍🦺Bài 02a")
   GiamDanDaoHam_Bai02a()

with tab4:
   st.header("🐈Bài 03")
   GiamDanDaoHam_Bai03()

with tab5:
   st.header("🐈‍⬛Bài 04")
   GiamDanDaoHam_Bai04()

with tab6:
   st.header("🦴Bài 05")
   GiamDanDaoHam_Bai05()

with tab7:
   st.header("🐠Temp")
   GiamDanDaoHam_Temp()