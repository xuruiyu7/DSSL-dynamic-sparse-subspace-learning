import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso,LinearRegression
import time
from scipy.stats import ortho_group
#import imageio
import os

import warnings
warnings.filterwarnings("ignore")

def cp_return(cp_vector):
    length=np.size(cp_vector)
    cp_temp=int(cp_vector[length-1])
    cp_list=[cp_temp]
    while cp_temp>0:
        cp_temp=int(cp_vector[cp_temp-1])
        cp_list.extend([cp_temp])
    cp_list=cp_list[::-1]
    return cp_list

def draw_cp(data,cp_list):
    plt.figure()
    plt.plot(data[:,0])
    for i in cp_list:
        plt.axvline(i)
    return 0

def Cost_select_y(data,t_left,t_right,lambda1,select_y_list):
    cost=0
    #dimen=np.size(data,1)
    len_data=t_right-t_left+1
    for i in select_y_list:
        data_temp=data[t_left-1:t_right,:].copy()
        data_y=data_temp[:,i].copy()
        data_x=data_temp.copy()
        data_x[:,i]=0
        lasso = Lasso(lambda1,fit_intercept=False,max_iter=1000)
        lasso.fit(data_x, data_y)
        coef=lasso.coef_
        data_y_estimate=np.dot(data_x,coef)
        cost+=np.sum((data_y_estimate-data_y)**2)/2+lambda1*len_data*np.linalg.norm(coef,ord=1)
        #cost+=max(dimen-len_data,0)*lambda1
    return cost

def miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list,delta=5):
    length_R=np.size(R)
    cost_vector=np.zeros(length_R)
    F_vector=np.zeros(length_R)
    for i in range(length_R):
        tau_temp=R[i]
        cost_vector[i]=Cost_select_y(data,tau_temp+1,tau,lambda1,select_y_list)
        F_vector[i]=F[tau_temp]+cost_vector[i]+lambda2
    F_tau=np.min(F_vector)
    F_vector_list=F_vector.tolist()
    cp_tau_index=F_vector_list.index(min(F_vector_list))
    cp_tau=R[cp_tau_index]

    while tau-cp_tau<delta:
        if cp_tau==0:
            break
        F_vector[cp_tau_index]=1e6
        F_tau=np.min(F_vector)
        if F_tau>=1e6:
            cp_tau=0
            F_tau=F[cp_tau]+Cost_select_y(data,cp_tau+1,tau,lambda1,select_y_list)+lambda2
            break
        F_vector_list=F_vector.tolist()
        cp_tau_index=F_vector_list.index(min(F_vector_list))
        cp_tau=R[cp_tau_index]       
    R_new=[]
    if tau<delta:
        R_new.extend([0])
    else:       
        for i in range(1,4):
            if tau-i*delta>=0:
                R_new.extend([tau-i*delta])
        for i in range(length_R):
            tau_temp=R[i]
            if tau_temp in R_new:
                continue
            F_tau_temp=F[tau_temp]+cost_vector[i]+K
            if F_tau_temp<F_tau:
                R_new.extend([tau_temp])
    return F_tau,cp_tau,R_new

def PELT_select_y(data,lambda1,lambda2,K,select_y_list,delta=5):
    length=np.shape(data)[0]
    #initialize
    F=np.zeros(length+1)
    cp=np.zeros(length)
    F[0]=-lambda2
    R=[0]
    #iterate
    for tau in range(1,length+1):
        F[tau],cp[tau-1],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list,delta)
        print(tau,cp[tau-1],len(R),F[tau])
        print(R)
    return cp

def PELT_select_y_with_F(data,lambda1,lambda2,K,select_y_list,delta=5):
    length=np.shape(data)[0]
    #initialize
    F=np.zeros(length+1)
    cp=np.zeros(length)
    F[0]=-lambda2
    R=[0]
    #iterate
    for tau in range(1,length+1):
        F[tau],cp[tau-1],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list,delta)
        print(tau,cp[tau-1],len(R),F[tau])
    return cp,F

def PELT_select_y_time(data,lambda1,lambda2,K,select_y_list,delta=5):
    length=np.shape(data)[0]
    #initialize
    F=np.zeros(length+1)
    cp=np.zeros(length)
    F[0]=-lambda2
    R=[0]
    #iterate
    t_list=[]
    R_list=[]
    begin_t=time.time()
    for tau in range(1,length+1):
        R_list.append(R)
        F[tau],cp[tau-1],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list,delta)
        print(tau,cp[tau-1],len(R),F[tau])
        t_list.extend([time.time()-begin_t])
    return cp,t_list,R_list

def OP_miniF_select_y(data,F,lambda1,lambda2,tau,R,K,select_y_list):
    length_R=np.size(R)
    cost_vector=np.zeros(length_R)
    F_vector=np.zeros(length_R)
    for i in range(length_R):
        tau_temp=R[i]
        cost_vector[i]=Cost_select_y(data,tau_temp,tau,lambda1,select_y_list)
        #print(Cost_select_y(data,tau_temp,tau,lambda1,select_y_list))
        F_vector[i]=F[tau_temp+1]+cost_vector[i]+lambda2
    
    F_tau=np.min(F_vector)
    F_vector_list=F_vector.tolist()
    cp_tau_index=F_vector_list.index(min(F_vector_list))
    cp_tau=R[cp_tau_index]
    R_new=[tau]
    R_new.extend(R)
    return F_tau,cp_tau,R_new


def OP_select_y_time(data,lambda1,lambda2,K,select_y_list):
    length=np.shape(data)[0]
    #initialize
    F=np.zeros(length)
    cp=np.zeros(length)
    F[0]=-lambda2
    R=[0]
    #iterate
    t_list=[]
    begin_t=time.time()
    for tau in range(length):
        if tau==0:
            continue
        F[tau+1],cp[tau],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,K,select_y_list)
        print(tau,cp[tau],len(R))
        t_list.extend([time.time()-begin_t])
    return cp,t_list

def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2

def basic_functions_plot(length):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    ax.scatter(range(32),basic_function_Bspline(32)[:,0],color='', marker='o', edgecolors='r')
    ax.plot(np.array(range(320))/10,basic_function_Bspline(320)[:,0],color='black')
    plt.xticks(fontsize=20)
    plt.yticks([0,.5,1,1.5,2,2.5],fontsize=15)
    plt.ylim([-0.2,2.5])
    plt.savefig('./plot/basic_functions/basic_signal_1.png',dpi=300)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    ax.scatter(range(32),basic_function_Bspline(32)[:,1],color='', marker='o', edgecolors='r')
    ax.plot(np.array(range(320))/10,basic_function_Bspline(320)[:,1],color='black')
    plt.xticks(fontsize=20)
    plt.yticks([0,.5,1,1.5,2,2.5],fontsize=15)
    plt.ylim([-0.2,2.5])
    plt.savefig('./plot/basic_functions/basic_signal_2.png',dpi=300)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    ax.scatter(range(32),basic_function_Bspline(32)[:,2],color='', marker='o', edgecolors='r')
    ax.plot(np.array(range(320))/10,basic_function_Bspline(320)[:,2],color='black')
    plt.xticks(fontsize=20)
    plt.yticks([0,.5,1,1.5,2,2.5],fontsize=15)
    plt.ylim([-0.2,2.5])
    plt.savefig('./plot/basic_functions/basic_signal_3.png',dpi=300)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    ax.scatter(range(32),basic_function_Fourier(32)[:,0],color='', marker='o', edgecolors='r')
    ax.plot(np.array(range(320))/10,basic_function_Fourier(320)[:,0],color='black')
    plt.xticks(fontsize=20)
    plt.yticks([-1,-.5,0,.5,1],fontsize=15)
    plt.savefig('./plot/basic_functions/basic_signal_4.png',dpi=300)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    ax.scatter(range(32),-basic_function_Fourier(32)[:,1],color='', marker='o', edgecolors='r')
    ax.plot(np.array(range(320))/10,-basic_function_Fourier(320)[:,1],color='black')
    plt.xticks(fontsize=20)
    plt.yticks([-1,-.5,0,.5,1],fontsize=15)
    plt.savefig('./plot/basic_functions/basic_signal_5.png',dpi=300)
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    ax.scatter(range(32),basic_function_Fourier(32)[:,2],color='', marker='o', edgecolors='r')
    ax.plot(np.array(range(320))/10,basic_function_Fourier(320)[:,2],color='black')
    plt.xticks(fontsize=20)
    plt.yticks([-1,-.5,0,.5,1],fontsize=15)
    plt.savefig('./plot/basic_functions/basic_signal_6.png',dpi=300)
    return 0

def basic_function_Bspline(length):
    bf=np.zeros((3,length))
    num=[0,7,14]
    nknots = 20   
    knots = np.linspace(0,1,nknots)
    for i in range(3):
        basis=np.random.random(1)[0]*10
        for j in range(length):
            bf[i,j]=B(j/length, 3, num[i], knots)*4
    return np.transpose(bf)
    
def basic_function_Fourier(length):
    bf=np.zeros((3,length))
    for i in range(3):
        basis=np.random.random(1)[0]*10
        for j in range(length):
            bf[i,j]=np.sin((i+1)*(j/length*2*np.pi+np.pi))
    return np.transpose(bf)    

def basic_function_Polynomials(length):
    bf=np.zeros((3,length))
    basis=np.random.random(1)[0]*10
    for j in range(length):
        j_s=j/(length-1)
        bf[0,j]=j_s*(1-j_s)*(0.25-j_s)
        bf[1,j]=j_s*(1-j_s)*(0.75-j_s)
        bf[2,j]=j_s*(1-j_s)*(0.25-j_s)*(0.75-j_s)*6
    return np.transpose(bf)*10

def simulation(length,n_signals):
    n_signal=int(n_signals/2)
    data=np.zeros((length,2*n_signal))
    tau_cp=[0,int(length/4),int(length/2),length]
    weight1=np.random.random((3,n_signal))-0.5
    weight2=np.random.random((3,n_signal))-0.5
    for k in range(3):
        weight1_new=np.random.random((3,n_signal))-0.5
        weight1_final=weight1_new*weight1/np.abs(weight1)*(-1)
        #weight=weight/weight.sum(axis=0)
        data[tau_cp[k]:tau_cp[k+1],0:n_signal]=np.dot(basic_function_Bspline(tau_cp[k+1]-tau_cp[k]),weight1_final)
        weight2_new=np.random.random((3,n_signal))-0.5
        weight2_final=weight2_new*weight2/np.abs(weight2)*(-1)
        #weight=weight/weight.sum(axis=0)
        data[tau_cp[k]:tau_cp[k+1],n_signal:2*n_signal]=np.dot(basic_function_Fourier(tau_cp[k+1]-tau_cp[k]),weight2_final)
    return data

def simulation_similar(length,n_signals):
    n_signal=int(n_signals/2)
    data=np.zeros((length,2*n_signal))
    tau_cp=[0,int(length/4),int(length/2),length]
    weight1=np.random.random((3,n_signal))-0.5
    weight2=np.random.random((3,n_signal))-0.5
    for k in range(3):
        weight1_new=np.random.random((3,3))-0.5
        weight1=np.dot(weight1_new,weight1)*2
        #weight=weight/weight.sum(axis=0)
        data[tau_cp[k]:tau_cp[k+1],0:n_signal]=np.dot(basic_function_Bspline(tau_cp[k+1]-tau_cp[k]),weight1)
        weight2_new=np.random.random((3,3))-0.5
        weight2=np.dot(weight2_new,weight2)*2
        #weight=weight/weight.sum(axis=0)
        data[tau_cp[k]:tau_cp[k+1],n_signal:2*n_signal]=np.dot(basic_function_Fourier(tau_cp[k+1]-tau_cp[k]),weight2)
    return data

def simulation_large(length,n_signals):
    n_signal=int(n_signals/2)
    data=np.zeros((length,2*n_signal))
    internel=int(length/10)
    tau_cp=[internel*i for i in range(11)]
    weight1=np.random.random((3,n_signal))-0.5
    weight2=np.random.random((3,n_signal))-0.5
    for k in range(10):
        weight1_new=np.random.random((3,n_signal))-0.5
        weight1_final=weight1_new*weight1/np.abs(weight1)*(-1)
        #weight=weight/weight.sum(axis=0)
        data[tau_cp[k]:tau_cp[k+1],0:n_signal]=np.dot(basic_function_Bspline(tau_cp[k+1]-tau_cp[k]),weight1_final)
        weight2_new=np.random.random((3,n_signal))-0.5
        weight2_final=weight2_new*weight2/np.abs(weight2)*(-1)
        #weight=weight/weight.sum(axis=0)
        data[tau_cp[k]:tau_cp[k+1],n_signal:2*n_signal]=np.dot(basic_function_Fourier(tau_cp[k+1]-tau_cp[k]),weight2_final)
    return data

def gen_correlation_matrix(n_signal,eta):
    d = n_signal
    beta = eta + (d-1)/2   
    P = np.zeros((d,d))
    S = np.eye(d)

    for k in range(1,d):
        beta = beta - 1/2;
        for i in range(k+1,d+1):
            P[k-1,i-1] = np.random.beta(beta,beta)
            P[k-1,i-1] = (P[k-1,i-1]-0.5)*2
            p = P[k-1,i-1]
            for l in range(k-1,0,-1):
                p = p * np.sqrt((1-P[l-1,i-1]**2)*(1-P[l-1,k-1]**2)) + P[l-1,i-1]*P[l-1,k-1]
            S[k-1,i-1] = p
            S[i-1,k-1] = p
    return S
    
def gen_correlation_matrix2(n_signal,betaparam):
    d = n_signal
    P = np.zeros((d,d))
    S = np.eye(d)

    for k in range(1,d):
        for i in range(k+1,d+1):
            P[k-1,i-1] = np.random.beta(betaparam,betaparam)
            P[k-1,i-1] = (P[k-1,i-1]-0.5)*2
            p = P[k-1,i-1]
            for l in range(k-1,0,-1):
                p = p * np.sqrt((1-P[l-1,i-1]**2)*(1-P[l-1,k-1]**2)) + P[l-1,i-1]*P[l-1,k-1]
            S[k-1,i-1] = p
            S[i-1,k-1] = p
    return S
    

def simulation_graph(length,n_signals):
    n_signal=int(n_signals/2)
    data=np.zeros((length,2*n_signal))
    tau_cp=[0,int(length/4),int(length/2),length]
    for k in range(3):
        weight=np.zeros((n_signals,n_signals))        
        eta=0.5
        weight1=gen_correlation_matrix2(n_signal,eta)
        weight2=gen_correlation_matrix2(n_signal,eta)
        weight[:n_signal,:n_signal]=weight1
        weight[n_signal:,n_signal:]=weight2
        mean=np.zeros(n_signals)

        data[tau_cp[k]:tau_cp[k+1],:]=np.random.multivariate_normal(mean, weight, (tau_cp[k+1]-tau_cp[k],), 'raise')*10
    return data




def create_gif(image_path, gif_name, duration):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    '''
    frames = []
    frames_list=os.listdir(image_path)
    for image_name in os.listdir(image_path):  
        print(image_name)
        frames.append(imageio.imread(image_path+'/'+image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return 0

def trace_plot(data,path):    ###plot
    signal_num=data.shape[1]
    for i in range(signal_num):
        plt.figure()
        plt.plot(data[:,i])
        plt.savefig(path+'/signal_%05d_trace.png'%(i+1))
    return 0

def trace_r_plot(data,path):    ###plot
    signal_num=data.shape[1]
    for i in range(signal_num):
        plt.figure()
        plt.plot(data[:,i])
        plt.savefig(path+'/signal_%05d_r_trace.png'%(i+1))
    return 0

def trace_plot_3d(data_arr,path): 
    signal_num=data_arr.shape[1]
    joint_num=int(signal_num/3)
    xmin=10
    ymin=10
    zmin=10
    xmax=-10
    ymax=-10
    zmax=-10
    for i in range(joint_num):
        xmin=min(np.min(data_arr[:,3*i]),xmin)
        xmax=max(np.max(data_arr[:,3*i]),xmax)
        ymin=min(np.min(data_arr[:,3*i+2]),ymin)
        ymax=max(np.max(data_arr[:,3*i+2]),ymax)
        zmin=min(np.min(data_arr[:,3*i+1]),zmin)
        zmax=max(np.max(data_arr[:,3*i+1]),zmax)
    dis_max=(max(max(xmax-xmin,ymax-ymin),zmax-zmin)+0.1)/2
    for j in range(data_arr.shape[0]):
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        for i in range(joint_num):
            ax.scatter(data_arr[j,3*i], data_arr[j,3*i+2], data_arr[j,3*i+1],marker='o',c='',edgecolors='b')  
            #ax.text(data_arr[j,3*i], data_arr[j,3*i+2], data_arr[j,3*i+1],str(i+1))
        for i,k in [[4,3],[3,9],[3,5],[3,2],[10,9],[10,11],[11,12],[5,6],[6,7],[7,8],[1,2],[1,17],[1,13],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]:
            ax.plot([data_arr[j,3*i-3], data_arr[j,3*k-3]],[data_arr[j,3*i+2-3],data_arr[j,3*k+2-3]],[data_arr[j,3*i+1-3], data_arr[j,3*k+1-3]],color='b')
        ax.set_aspect("equal")
        ax.set_xlabel('X')
        ax.set_xlim((xmin+xmax)/2-dis_max,(xmin+xmax)/2+dis_max)
        ax.set_ylabel('Y')
        ax.set_ylim((ymin+ymax)/2-dis_max,(ymin+ymax)/2+dis_max)
        ax.set_zlabel('Z')
        ax.set_zlim((zmin+zmax)/2-dis_max,(zmin+zmax)/2+dis_max)
        ax.axis('off')        
        plt.savefig(path+'/signal2_%05d_trace.png'%(j+1),dpi=300)
    return 0
        
def load_data(data_path):
    data=pd.read_csv(data_path,sep=' ',header =None)
    data.drop([0],axis=1,inplace=True)
    data=data[~data[1].isin([0])]
    signal_num=data.shape[1]
    joint_num=int(signal_num/4)
    delete_columns=[4*i+4 for i in range(joint_num)]
    data=data.drop(delete_columns,axis=1)
    signal_num=data.shape[1]
    data.columns=range(signal_num)
    data_arr=np.array(data)
    print("data loading done.")
    return data_arr


def determine_parameters(data_past,n_signals,cp_list,K,delta=5):
    num_data=np.shape(data_past)[0]
    len_data=cp_list[-1]
    cost_t_list=[]
    lambda1_list=[]
    lambda1=1e-5
    is_zero=0
    while is_zero==0:
        lambda1=lambda1*10
        lambda1_list.extend([lambda1])
        cost=0
        cost1=0
        cost2=0
        cost_t=0
        for j in range(num_data):
            data_noise=data_past[j,:,:]
            for k in range(len(cp_list)-1):
                data_noise_temp=data_noise[cp_list[k]:cp_list[k+1],:]
                for i in range(n_signals):
                    data_temp=data_noise_temp
                    data_y=data_temp[:,i].copy()
                    data_x=data_temp.copy()
                    data_x[:,i]=0
                    #lasso = Lasso(lambda1/len(data_y)/2,fit_intercept=False,max_iter=10000)
                    lasso = Lasso(lambda1,fit_intercept=False,max_iter=10000)
                    #lasso = LinearRegression()
                    lasso.fit(data_x, data_y)
                    coef=lasso.coef_
                    cost=cost+np.linalg.norm(coef,ord=1)
                    data_y_estimate_temp=np.dot(data_x,coef)
                    cost1=cost1+np.sum((data_y_estimate_temp-data_y)**2)
                    cost2=cost2+np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
                    #cost_t+=np.sum((data_y_estimate_temp-data_y)**2)/0.05/0.05++3*np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
                    #cost_t+=len(data_y_estimate_temp)*np.log(np.sum((data_y_estimate_temp-data_y)**2)/len(data_y_estimate_temp))\
                            #+3*np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+cost2
        print(lambda1,np.log(cost1/n_signals/num_data/len_data),cost2,cost_t)
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])
        #print(cost)
        if cost/num_data/(len(cp_list)-1)/n_signals<1e-4:
            is_zero=1
            break    
    
    lambda1_max=lambda1  
    print(lambda1_max)

    best_index=cost_t_list.index(min(cost_t_list))
    lambda_best=lambda1_list[best_index]
    print(lambda_best)

    lambda1_list=[lambda_best/10*i for i in range(11)]
    lambda1_list.extend([lambda_best*(i+2) for i in range(9)])

    cost_t_list=[]
    for index_lambda1 in range(len(lambda1_list)):
        coef_matrix_large_t=np.zeros((n_signals,n_signals))
        cost1=0
        cost2=0
        cost_t=0
        for j in range(num_data):
            data_noise=data_past[j,:,:]
            for k in range(len(cp_list)-1):
                data_noise_temp=data_noise[cp_list[k]:cp_list[k+1],:]
                for i in range(n_signals):
                    coef_matrix=np.zeros((n_signals+3))
                    data_temp=data_noise_temp
                    data_y=data_temp[:,i].copy()
                    data_x=data_temp.copy()
                    data_x[:,i]=0
                    #lasso = Lasso(lambda1_list[index_lambda1]/len(data_y)/2,fit_intercept=False,max_iter=10000)
                    lasso = Lasso(lambda1_list[index_lambda1],fit_intercept=False,max_iter=10000)
                    lasso.fit(data_x, data_y)
                    coef=lasso.coef_
                    r_square=lasso.score(data_x,data_y)
                    data_y_estimate_temp=np.dot(data_x,coef)
                    cost1=cost1+np.sum((data_y_estimate_temp-data_y)**2)
                    coef_matrix[0:n_signals]=coef
                    coef_matrix[n_signals]=0
                    coef_matrix[n_signals+1]=cost1
                    coef_matrix[n_signals+2]=r_square
                    coef_matrix_large_t[:,i]=coef
                    cost2=cost2+np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
                    #cost_t+=np.sum((data_y_estimate_temp-data_y)**2)/0.05/0.05++3*np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
                    #cost_t+=len(data_y_estimate_temp)*np.log(np.sum((data_y_estimate_temp-data_y)**2)/len(data_y_estimate_temp))\
                            #+3*np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])
        print(lambda1_list[index_lambda1],np.log(cost1/n_signals/num_data/len_data),cost2,cost_t)


    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda_best=lambda1_list[best_index]
    print(lambda_best)

    if best_index==0:
        lambda1_list=[lambda1_list[best_index]+(lambda1_list[best_index+1]-lambda1_list[best_index])/10*i for i in range(11)]
    elif best_index==len(lambda1_list)-1:
        lambda1_list=[lambda1_list[best_index-1]+(lambda1_list[best_index]-lambda1_list[best_index-1])/10*i for i in range(11)]
    else:
        lambda1_list=[lambda1_list[best_index-1]+(lambda1_list[best_index+1]-lambda1_list[best_index-1])/10*i for i in range(11)]
    cost_t_list=[]
    for index_lambda1 in range(11):
        coef_matrix_large_t=np.zeros((n_signals,n_signals))
        cost1=0
        cost2=0
        cost_t=0
        for j in range(num_data):
            data_noise=data_past[j,:,:]
            for k in range(len(cp_list)-1):
                data_noise_temp=data_noise[cp_list[k]:cp_list[k+1],:]
                for i in range(n_signals):
                    coef_matrix=np.zeros((n_signals+3))
                    data_temp=data_noise_temp
                    data_y=data_temp[:,i].copy()
                    data_x=data_temp.copy()
                    data_x[:,i]=0
                    #lasso = Lasso(lambda1_list[index_lambda1]/len(data_y)/2,fit_intercept=False,max_iter=10000)
                    lasso = Lasso(lambda1_list[index_lambda1],fit_intercept=False,max_iter=10000)
                    lasso.fit(data_x, data_y)
                    coef=lasso.coef_
                    r_square=lasso.score(data_x,data_y)
                    data_y_estimate_temp=np.dot(data_x,coef)
                    cost1=cost1+np.sum((data_y_estimate_temp-data_y)**2)
                    coef_matrix[0:n_signals]=coef
                    coef_matrix[n_signals]=0
                    coef_matrix[n_signals+1]=cost1
                    coef_matrix[n_signals+2]=r_square
                    coef_matrix_large_t[:,i]=coef
                    cost2=cost2+np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
                    #cost_t+=np.sum((data_y_estimate_temp-data_y)**2)/0.05/0.05++3*np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
                    #cost_t+=len(data_y_estimate_temp)*np.log(np.sum((data_y_estimate_temp-data_y)**2)/len(data_y_estimate_temp))\
                            #+3*np.log(len(data_y_estimate_temp))*np.size(np.nonzero(coef),1)
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])
        print(lambda1_list[index_lambda1],np.log(cost1/n_signals/num_data/len_data),cost2,cost_t)


    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda1=lambda1_list[best_index]
    print(lambda1)

    select_y_list=[i for i in range(n_signals)]
    cp_ture=np.zeros(cp_list[-1])
    for i in range(len(cp_list)-1):
        cp_ture[cp_list[i]:cp_list[i+1]]=cp_list[i]    
    
    lambda2=1e-3
    #lambda2=-10
    is_zero=0
    lambda2_list=[]
    cp_est_list=[]
    cost_list=[]
    while is_zero==0:
        lambda2=lambda2*10
        lambda2_list.extend([lambda2])
        cost=0
        for j in range(num_data):
            data_noise=data_past[j,:,:]
            cp=PELT_select_y(data_noise,lambda1,lambda2,lambda2/1.5,select_y_list)
            cp_est=cp.tolist()
            cp_est_list.append(cp_est)
            cp_est=np.array(cp_est)
            cp_diff=cp_ture-cp_est
            cp_diff[np.where((cp_diff>-delta)&(cp_diff<delta))]=0
            cost=cost+np.size(np.nonzero(cp_diff),1)
        #if len(cost_list)>0:
        #    if cost>=cost_list[-1]:
        #        cost_list.extend([cost])
        #        is_zero=1
        #        break   
        cost_list.extend([cost])
        #print(cost)
        if np.linalg.norm(cp_est,ord=1)/cp_list[-1]<1:
            is_zero=1
            break    
    lambda2_max=lambda2 
    print(lambda2_max)
    
    best_index=cost_list.index(min(cost_list))
    lambda2_best=lambda2_list[best_index]
    cost_positive=cost_list[best_index]
    
    lambda2_list=[lambda2_best/10*i for i in range(11)]
    lambda2_list.extend([lambda2_best*(i+2) for i in range(9)])
    cp_est_list=[]
    cost_list=[]
    for index_lambda2 in range(len(lambda2_list)):
        cost=0
        for j in range(num_data):
            data_noise=data_past[j,:,:]
            cp=PELT_select_y(data_noise,lambda1,lambda2_list[index_lambda2],lambda2_list[index_lambda2]/1.5,select_y_list)
            cp_est=cp.tolist()
            cp_est_list.append(cp_est)
            cp_est=np.array(cp_est)
            cp_diff=cp_ture-cp_est
            cp_diff[np.where((cp_diff>-delta)&(cp_diff<delta))]=0
            cost=cost+np.size(np.nonzero(cp_diff),1)
        cost_list.extend([cost])
    #print(cost_t_list)
    best_index=np.where(np.array(cost_list)==min(cost_list))
    lambda2_best=np.max(np.array(lambda2_list)[best_index])
    lambda2=lambda2_best
    if len(best_index)>1:
        return lambda1,lambda2
    else:
        best_index=best_index[0][0]
        if best_index==0:
            lambda2_list=[lambda2_list[best_index]+(lambda2_list[best_index+1]-lambda2_list[best_index])/10*i for i in range(11)]
        elif best_index==len(lambda2_list)-1:
            lambda2_list=[lambda2_list[best_index-1]+(lambda2_list[best_index]-lambda2_list[best_index-1])/10*i for i in range(11)]
        else:
            lambda2_list=[lambda2_list[best_index-1]+(lambda2_list[best_index+1]-lambda2_list[best_index-1])/10*i for i in range(11)]
        cp_est_list=[]
        cost_list=[]
        for index_lambda2 in range(len(lambda2_list)):
            cost=0
            for j in range(num_data):
                data_noise=data_past[j,:,:]
                cp=PELT_select_y(data_noise,lambda1,lambda2_list[index_lambda2],lambda2_list[index_lambda2]/1.5,select_y_list)
                cp_est=cp.tolist()
                cp_est_list.append(cp_est)
                cp_est=np.array(cp_est)
                cp_diff=cp_ture-cp_est
                cp_diff[np.where((cp_diff>-delta)&(cp_diff<delta))]=0
                cost=cost+np.size(np.nonzero(cp_diff),1)
            cost_list.extend([cost])
        #print(cost_t_list)
        best_index=np.where(np.array(cost_list)<min(cost_list)+num_data*delta)
        lambda2_best=np.mean(np.array(lambda2_list)[best_index])
        lambda2=lambda2_best
        return lambda1,lambda2
'''
sigma2=0.05
lambda1=0.016
lambda2=6
n_signals=40
K=lambda2/10
n_length=128
n_experiments=100
np.random.seed(0)
cp_list=[]
    
data_patch=np.zeros((n_experiments,n_length,n_signals))
data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
np.random.seed(0)
data=simulation(n_length,n_signals)
data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
select_y_list=[i for i in range(n_signals)]
cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
'''

