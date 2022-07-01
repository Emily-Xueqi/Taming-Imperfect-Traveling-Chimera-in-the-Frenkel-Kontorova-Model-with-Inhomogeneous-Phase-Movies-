# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:31:04 2022

@author: lxq
"""


# from statsmodels.tsa.stattools import grangercausalitytests
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from timeit import default_timer as timer
# from sklearn.metrics.pairwise import cosine_similarity
# from minepy import MINE

@nb.jit(nopython=True)
def equation(t, X0, φ_n):
    m=1.0; g=1.0; gamma=0.75; taop=0.7155 
    tao=0.4; K=0.5; omega=0.25; L=1. 
  
    x, dx = X0 
    # print(x.shape)
    x = np.append(x[0],x)
    x = np.append(x,x[-1])
    ddx = (-m*g*L*np.sin(x[1:N+1])+ taop+ tao*np.sin(omega*t+ φ_n)+ \
        K*(x[2:N+2] + x[0:N] -2*x[1:N+1])-gamma*dx)/(m*L**2)

    return np.vstack((dx, ddx)), K*(x[2:N+2] + x[0:N] -2*x[1:N+1])/(m*L**2)

def solt_points(equation,X0, φ_n): 
    @nb.jit(nopython=True)
    def jit_fun(equation=equation,X0=X0, φ_n=φ_n):
        omega=0.25
        T = 2*np.pi/omega
        h = T/250
        trans = h*10000
        ends  = h*20000
        Ttrans = np.arange(0,trans, h)
        T = np.arange(trans,ends, h)
        sol = []; sol_theta=[]
        for t in Ttrans:
            K1,_ = equation(t, X0, φ_n)
            K2,_ = equation(t + h/2., X0 + h * K1/2., φ_n)
            K3,_ = equation(t + h/2., X0 + h * K2/2., φ_n)
            K4,_ = equation(t + h, X0 + h * K3, φ_n)
            X0 = X0 + h * (K1 + 2.* K2 + 2. * K3 + K4)/6.         
        for t in T:
            K1,_ = equation(t, X0, φ_n)
            K2,_ = equation(t + h/2., X0 + h * K1/2., φ_n)
            K3,_ = equation(t + h/2., X0 + h * K2/2., φ_n)
            K4,_ = equation(t + h, X0 + h * K3, φ_n)
            X0 = X0 + h * (K1 + 2.* K2 + 2. * K3 + K4)/6. 
            sol.append(X0[1])
            sol_theta.append(X0[0])          
        return sol, sol_theta
    return jit_fun

def plot(solt,title):
    N, I = solt.shape
    Xi, Yi = np.meshgrid(np.arange(I)*h, np.arange(1,N+1))    
    fig,ax = plt.subplots(figsize=(8, 3),dpi=300) 
    pcm = plt.cm.get_cmap('jet') #
    cont = ax.contourf(Xi, Yi, solt, 100,cmap=pcm)
    font = {'family':'Times New Roman', 'weight':'normal', 'size': 20} 
    plt.ylabel('node index',font,rotation=90)
    plt.xlabel('time',font,horizontalalignment='center',rotation=0)
    plt.yticks([1, 32, 64, 96, 128], size=20,weight='bold', family='Times New Roman')
    plt.xticks([0, 200, 400, 600, 800, 1000], size=20,weight='bold', family='Times New Roman')
    plt.title(title)
    cb = plt.colorbar(cont,location="right")
    cb.set_ticks([-0.90,  0.00,  0.90,  1.80,  2.70])
    cb.set_label("$\.θ$", fontdict = font, rotation=0)
    labels = cb.ax.yaxis.get_ticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    cb.ax.tick_params(labelsize="18")
    
    
def snapshot(UU):
    plt.figure(dpi=300,figsize=(8, 6))  #设置图像的大小  
    S = np.arange(1, N+1)
    plt.scatter(S, UU, c="b")
    plt.plot(S, UU, c="grey")
    
    plt.tick_params(labelsize=18) #刻度字体大小13
    plt.xlim(1, N)
    plt.ylim(-1, 3.4)
    font = {'family':'Times New Roman', 'weight':'normal', 'size': 22} 
    # plt.title('(b)',font)
    plt.yticks([-1.0, 0.0, 1.0, 2.0, 3],size=20,weight='normal', family='Times New Roman')
    plt.xticks([1, 32, 64, 96, 128],size=20,weight='normal', family='Times New Roman')
    plt.xlabel('node index',font, rotation=0)
    plt.ylabel('$\.θ$',font, rotation=0, labelpad=8.5)
    # plt.show()


if __name__ == '__main__':
    begin =timer() 
    N = 128
    title = " "
    
    omega=0.25
    T = 2*np.pi/omega
    h = T/250
    trans = h*10000
    ends  = h*20000  
    
    #初值条件
    b=0.5
    X0_0 = np.arange(0, N*b, b)-(N-1)/2.*b
    X0_1 = np.zeros(N)
    X0 = np.vstack((X0_0, X0_1))
    
    k=0.0
    np.random.seed(38)
    φ_n  = np.random.uniform(low=-k*np.pi, high=k*np.pi, size=N)
    
    jit_fun = solt_points(equation,X0, φ_n)
    sol, sol_theta = jit_fun()
    solt = np.vstack(sol).T    #dtheta.shape=(N,T)=(50,5001)
    sol_thetat = np.vstack(sol_theta).T
    
    
    filenames = []
    num = 0
    for i in range(solt.shape[1]):
        if i%10 ==0:
            # print(5)
            snapshot(solt[:,i])
            num += 1
            filename = f'{num}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.close()
        
    # 生成gif
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # 删除40张折线图
    for filename in set(filenames):
        os.remove(filename)
    
    
    plot(solt,title)
    end =timer()    
    print("程序运行时间:" ,(end - begin)/60,'min')    
    
    
    # plot(solt,title)   #(50,5001)
    # snapshot(solt[:,-2], solt[:,-26], solt[:,-345], solt[:,-349])
    # phase_t1(solt[20,:], solt[27,:], solt[21,:], solt[24,:], solt[25,:])
