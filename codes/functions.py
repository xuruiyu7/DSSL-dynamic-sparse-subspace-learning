import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso,LinearRegression
import time
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
    for i in select_y_list:
        data_temp=data[t_left:t_right,:].copy()
        data_y=data_temp[:,i].copy()
        data_x=data_temp.copy()
        data_x[:,i]=0
        #N_temp=len(data_x)
        #lasso = Lasso(lambda1/N_temp/2,fit_intercept=False,max_iter=10000)
        lasso = Lasso(lambda1,fit_intercept=False,max_iter=10000)
        lasso.fit(data_x, data_y)
        coef=lasso.coef_
        data_y_estimate=np.dot(data_x,coef)
        cost=cost+np.sum((data_y_estimate-data_y)**2)+lambda1*np.linalg.norm(coef,ord=1)
    return cost

def miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list,delta=5):
    length_R=np.size(R)
    cost_vector=np.zeros(length_R)
    F_vector=np.zeros(length_R)
    for i in range(length_R):
        tau_temp=R[i]
        cost_vector[i]=Cost_select_y(data,tau_temp,tau+1,lambda1,select_y_list)
        #print(Cost_select_y(data,tau_temp,tau,lambda1,select_y_list))
        F_vector[i]=F[tau_temp]+cost_vector[i]+lambda2
    F_tau=np.min(F_vector)
    F_vector_list=F_vector.tolist()
    cp_tau_index=F_vector_list.index(min(F_vector_list))
    cp_tau=R[cp_tau_index]

    while tau-cp_tau<delta:
        F_vector[cp_tau_index]=1e4
        F_tau=np.min(F_vector)
        if F_tau>=1e4:
            cp_tau=0
            F_tau=F[cp_tau]+Cost_select_y(data,cp_tau,tau+1,lambda1,select_y_list)+lambda2
            break
        F_vector_list=F_vector.tolist()
        cp_tau_index=F_vector_list.index(min(F_vector_list))
        cp_tau=R[cp_tau_index]       
#    if tau-cp_tau<50:
#        cp_tau=int(cp[tau-1])
#        F_tau=F[cp_tau]+Cost_select_y(data,cp_tau,tau+1,lambda1,select_y_list)+lambda2
    R_new=[tau+1]
    for i in range(length_R):
        tau_temp=R[i]
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
    for tau in range(length):
        F[tau+1],cp[tau],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list,delta)
        print(tau,cp[tau],len(R),F[tau+1])
    return cp

def PELT_select_y_with_F(data,lambda1,lambda2,K,select_y_list):
    length=np.shape(data)[0]
    #initialize
    F=np.zeros(length+1)
    cp=np.zeros(length)
    F[0]=-lambda2
    R=[0]
    #iterate
    for tau in range(length):
        F[tau+1],cp[tau],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list)
        print(tau,cp[tau],len(R),F[tau+1])
    return cp,F

def PELT_select_y_time(data,lambda1,lambda2,K,select_y_list):
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
    for tau in range(length):
        R_list.append(R)
        F[tau+1],cp[tau],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list)
        print(tau,cp[tau],len(R),F[tau+1])
        t_list.extend([time.time()-begin_t])
    return cp,t_list,R_list

def PELT_select_y_fit(data,lambda1,lambda2,K,select_y_list):
    length=np.shape(data)[0]
    n_signals=np.shape(data)[1]
    #initialize
    data_fit=np.zeros((length,length,n_signals))
    F=np.zeros(length+1)
    cp=np.zeros(length)
    F[0]=-lambda2
    R=[0]
    #iterate
    t_list=[]
    R_list=[]
    begin_t=time.time()
    for tau in range(length):
        R_list.append(R)
        F[tau+1],cp[tau],R=miniF_select_y(data,F,lambda1,lambda2,tau,R,cp,K,select_y_list)
        print(tau,cp[tau],len(R),F[tau+1])
        t_list.extend([time.time()-begin_t])
        
        cp_list=cp_return(cp[:tau+1])
        cp_list.extend([tau+1])
        for i in range(n_signals):
            for j in range(len(cp_list)-1):
                data_temp=data[cp_list[j]:cp_list[j+1],:]
                data_y=data_temp[:,i].copy()
                data_x=data_temp.copy()
                data_x[:,i]=0
                lasso = Lasso(lambda1,fit_intercept=False)
                lasso.fit(data_x, data_y)
                coef=lasso.coef_
                data_y_estimate_temp=np.dot(data_x,coef)
                data_fit[tau,cp_list[j]:cp_list[j+1],i]=data_y_estimate_temp
    return cp,t_list,R_list,data_fit

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


def determine_parameters(data_past,n_signals,cp_list,K):
    num_data=np.shape(data_past)[0]
    len_data=cp_list[-1]
    
    lambda1=1e-8
    is_zero=0
    while is_zero==0:
        lambda1=lambda1*2
        cost=0
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
        #print(cost)
        if cost/num_data/(len(cp_list)-1)/n_signals<1e-4:
            is_zero=1
            break    
    
    lambda1_max=lambda1  
    print(lambda1_max)

    lambda1_list=[lambda1_max/10*i for i in range(11)]
    cost_t_list=[]
    cost_BIC_list=[]
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
                    cost2=cost2+np.size(np.nonzero(coef),1)

        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+np.log(n_signals*num_data*len_data)*cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])
    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda_best=lambda1_list[best_index]
    print(lambda_best)

    if best_index==0:
        lambda1_list=[lambda1_list[best_index]+(lambda1_list[best_index+1]-lambda1_list[best_index])/10*i for i in range(11)]
    elif best_index==10:
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
                    cost2=cost2+np.size(np.nonzero(coef),1)
                    
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+np.log(n_signals*num_data*len_data)*cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])

    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda_best=lambda1_list[best_index]
    print(lambda_best)

    if best_index==0:
        lambda1_list=[lambda1_list[best_index]+(lambda1_list[best_index+1]-lambda1_list[best_index])/10*i for i in range(11)]
    elif best_index==10:
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
                    cost2=cost2+np.size(np.nonzero(coef),1)
                    
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+np.log(n_signals*num_data*len_data)*cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])

    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda_best=lambda1_list[best_index]

    if best_index==0:
        lambda1_list=[lambda1_list[best_index]+(lambda1_list[best_index+1]-lambda1_list[best_index])/10*i for i in range(11)]
    elif best_index==10:
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
                    cost2=cost2+np.size(np.nonzero(coef),1)
                    
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+np.log(n_signals*num_data*len_data)*cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])

    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda1_best=lambda1_list[best_index]

    if best_index==0:
        lambda1_list=[lambda1_list[best_index]+(lambda1_list[best_index+1]-lambda1_list[best_index])/10*i for i in range(11)]
    elif best_index==10:
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
                    cost2=cost2+np.size(np.nonzero(coef),1)
                    
        cost_t=n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+np.log(n_signals*num_data*len_data)*cost2
        #cost_t_list.extend([cost_t])
        cost_t_list.extend([cost_t])

    #print(cost_t_list)
    best_index=cost_t_list.index(min(cost_t_list))
    lambda1_best=lambda1_list[best_index]
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
            cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
            cp_est=cp.tolist()
            cp_est_list.append(cp_est)
            cp_est=np.array(cp_est)
            cp_diff=cp_ture-cp_est
            cp_diff[np.where((cp_diff>-5)&(cp_diff<5))]=0
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
    
    if best_index!=0:
        lambda2_list=[lambda2_best/10*i for i in range(11)]
        lambda2_list.extend([lambda2_best*(i+2) for i in range(9)])
        cp_est_list=[]
        cost_list=[]
        for index_lambda2 in range(len(lambda2_list)):
            cost=0
            for j in range(num_data):
                data_noise=data_past[j,:,:]
                cp=PELT_select_y(data_noise,lambda1,lambda2_list[index_lambda2],K,select_y_list)
                cp_est=cp.tolist()
                cp_est_list.append(cp_est)
                cp_est=np.array(cp_est)
                cp_diff=cp_ture-cp_est
                cp_diff[np.where((cp_diff>-5)&(cp_diff<5))]=0
                cost=cost+np.size(np.nonzero(cp_diff),1)
            cost_list.extend([cost])
        #print(cost_t_list)
        best_index=cost_list.index(min(cost_list))
        lambda2_best=lambda2_list[best_index]
        lambda2=lambda2_best
    else:
        lambda2=-1e-3
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
                cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
                cp_est=cp.tolist()
                cp_est_list.append(cp_est)
                cp_est=np.array(cp_est)
                cp_diff=cp_ture-cp_est
                cp_diff[np.where((cp_diff>-5)&(cp_diff<5))]=0
                cost=cost+np.size(np.nonzero(cp_diff),1)
            if len(cost_list)>0:
                if cost>=cost_list[-1]:
                    cost_list.extend([cost])
                    is_zero=1
                    break   
            cost_list.extend([cost])
            #print(cost)
        lambda2_min=lambda2 
        print(lambda2_min)
        lambda2=lambda2_min/10
        
        lambda2_list=[lambda2/10*i for i in range(11)]
        lambda2_list.extend([lambda2*(i+2) for i in range(9)])
        cp_est_list=[]
        cost_list=[]
        for index_lambda2 in range(len(lambda2_list)):
            cost=0
            for j in range(num_data):
                data_noise=data_past[j,:,:]
                cp=PELT_select_y(data_noise,lambda1,lambda2_list[index_lambda2],K,select_y_list)
                cp_est=cp.tolist()
                cp_est_list.append(cp_est)
                cp_est=np.array(cp_est)
                cp_diff=cp_ture-cp_est
                cp_diff[np.where((cp_diff>-5)&(cp_diff<5))]=0
                cost=cost+np.size(np.nonzero(cp_diff),1)
            cost_list.extend([cost])
        #print(cost_t_list)
        best_index=cost_list.index(min(cost_list))
        lambda2_best=lambda2_list[best_index]
        lambda2=lambda2_best   
    return lambda1,lambda2
