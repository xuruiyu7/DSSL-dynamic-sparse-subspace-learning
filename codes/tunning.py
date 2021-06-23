from functions import *
import numpy as np

def cost_tunning(data_past,lambda1,lambda2,n_signals,K):
    num_data=np.shape(data_past)[0]
    len_data=np.shape(data_past)[1]

    coef_matrix_large_t=np.zeros((n_signals,n_signals))
    cost_t=0

    for j in range(num_data):
        cost1=0
        cost2=0
        data_noise=data_past[j,:,:]
        cp=PELT_select_y(data_noise,lambda1,lambda2,K,np.arange(0,n_signals))
        cp_list=cp_return(cp)
        cp_list.extend([len_data])
        
        for k in range(len(cp_list)-1):
            data_noise_temp=data_noise[cp_list[k]:cp_list[k+1],:]
            for i in range(n_signals):
                coef_matrix=np.zeros((n_signals+3))
                data_temp=data_noise_temp
                data_y=data_temp[:,i].copy()
                data_x=data_temp.copy()
                data_x[:,i]=0
                #lasso = Lasso(lambda1_list[index_lambda1]/len(data_y)/2,fit_intercept=False,max_iter=10000)
                lasso = Lasso(lambda1,fit_intercept=False,max_iter=10000)
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
                #coef[(coef<1e-3)&(coef>-1e-3)]=0
                cost2=cost2+np.size(np.nonzero(coef),1)
        cost_t=cost_t+n_signals*num_data*len_data*np.log(cost1/n_signals/num_data/len_data)+3*np.log(n_signals*num_data*len_data)*cost2
    return cp,cost_t
            
def determine_parameters_unsupervised(data_past,n_signals,K):
    num_data=np.shape(data_past)[0]
    len_data=np.shape(data_past)[1]

    lambda1=1e-8
    is_zero=0
    while is_zero==0:
        lambda1=lambda1*2
        cost=0
        for j in range(num_data):
            data_noise=data_past[j,:,:]
            data_noise_temp=data_noise
            for i in range(n_signals):
                data_temp=data_noise_temp
                data_y=data_temp[:,i].copy()
                data_x=data_temp.copy()
                data_x[:,i]=0
                #lasso = Lasso(lambda1/len(data_y)/2,fit_intercept=False,max_iter=10000)
                lasso = Lasso(lambda1,fit_intercept=False,max_iter=10000)
                lasso.fit(data_x, data_y)
                coef=lasso.coef_
                cost=cost+np.linalg.norm(coef,ord=1)
        #print(cost)
        if cost/num_data/n_signals<1e-4:
            is_zero=1
            break    
    lambda1_max=lambda1  
    print(lambda1_max)  

    cost_list_for_lambda1=[]
    lambda12_list=[]
    lambda1=0.0001
    while(lambda1<lambda1_max):
        lambda1=lambda1*10
        lambda2=1e-3
        is_zero=0
        lambda2_list=[]
        cp_est_list=[]
        cost_list=[]
        while is_zero==0:
            lambda2=lambda2*10
            lambda2_list.extend([lambda2])
            cp_est,cost=cost_tunning(data_past,lambda1,lambda2,n_signals,K)
            if len(cost_list)>0:
                if cost>cost_list[-1]:
                    cost_list.extend([cost])
                    is_zero=1
                    break   
            cost_list.extend([cost])
            #print(cost)
            if np.linalg.norm(cp_est,ord=1)/len_data<1:
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
            cp_est,cost=cost_tunning(data_past,lambda1,lambda2_list[index_lambda2],n_signals,K)
            if len(cost_list)>0:
                if cost>cost_list[-1]:
                    cost_list.extend([cost])
                    is_zero=1
                    break   
            cost_list.extend([cost])
            if np.linalg.norm(cp_est,ord=1)/len_data<1:
                is_zero=1
                break    

        #print(cost_t_list)
        best_index=cost_list.index(min(cost_list))
        lambda2_best=lambda2_list[best_index]
        cost_best=min(cost_list)
        cost_list_for_lambda1.extend([cost_best])
        lambda12_list.append([lambda1,lambda2_best])
        
    best_index=cost_list_for_lambda1.index(min(cost_list_for_lambda1))
    lambda12=lambda12_list[best_index]
    lambda1_best=lambda12[0]
    lambda2_best=lambda12[1]
    #lambda1_best=0.01
    #lambda2_best=2
    
    lambda1_list=[lambda1_best/10*(i+1) for i in range(10)]
    lambda1_list.extend([lambda1_best*(i+2) for i in range(9)])
    cp_est_list=[]
    cost_list=[]
    for index_lambda1 in range(len(lambda1_list)):
        cp_est,cost=cost_tunning(data_past,lambda1_list[index_lambda1],lambda2_best,n_signals,K)
        if len(cost_list)>0:
            if cost>cost_list[-1]:
                cost_list.extend([cost])
                is_zero=1
                break   
        cost_list.extend([cost])
        if np.linalg.norm(cp_est,ord=1)/len_data<1:
            is_zero=1
            break    
    best_index=cost_list.index(min(cost_list))
    lambda1_best=lambda1_list[best_index]
    cost_best=min(cost_list)

    lambda2_list=[lambda2_best/10*(i+1) for i in range(10)]
    lambda2_list.extend([lambda2_best*(i+2) for i in range(9)])
    cp_est_list=[]
    cost_list=[]
    for index_lambda2 in range(len(lambda2_list)):
        cp_est,cost=cost_tunning(data_past,lambda1_best,lambda2_list[index_lambda2],n_signals,K)
        if len(cost_list)>0:
            if cost>cost_list[-1]:
                cost_list.extend([cost])
                is_zero=1
                break   
        cost_list.extend([cost])
        if np.linalg.norm(cp_est,ord=1)/len_data<1:
            is_zero=1
            break    

    #print(cost_t_list)
    best_index=cost_list.index(min(cost_list))
    lambda2_best=lambda2_list[best_index]
    cost_best=min(cost_list)
    return lambda1_best,lambda2_best

    
