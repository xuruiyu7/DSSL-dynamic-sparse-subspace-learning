from functions import *
from tunning import *
import numpy as np                  
import scipy.io as sio

if __name__ == '__main__':
#############################################################################
    ###parameter tunning
    '''
    num_data=1
    n_signals=40
    data_past=np.zeros((num_data,128,n_signals))
    np.random.seed(0)
    for j in range(num_data):
        data=simulation(128,n_signals)
        data_noise=data+np.random.normal(size=(128,n_signals))*.05
        data_past[j,:,:]=data_noise
        
    cp_list=[0,32,64,128]
    K=0
    lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K)
    '''
#############################################################################
    data=np.zeros((128,40))
    np.random.seed(0)
    data[0,:]=np.random.normal(size=(1,40))
    data[1,:]=np.random.normal(size=(1,40))
    for i in range(2,32):
        data[i,:]=np.append(np.diff(data[i-1,:]),data[i-1,39])*.3+np.append(np.diff(data[i-2,:]),data[i-2,39])*-.3+np.random.normal(size=(1,40))*.05
    for i in range(32,64):
        data[i,:]=np.append(np.diff(data[i-1,:]),data[i-1,39])*-.3+np.append(np.diff(data[i-2,:]),data[i-2,39])*.3+np.random.normal(size=(1,40))*.05
    for i in range(64,128):
        data[i,:]=np.append(np.diff(data[i-1,:]),data[i-1,39])*.3+np.append(np.diff(data[i-2,:]),data[i-2,39])*-.3+np.random.normal(size=(1,40))*.05
    K=0
    lambda1=0.01
    lambda2=2
    n_length=128
    cp=PELT_select_y(data,lambda1,lambda2,K,select_y_list)
    plt.plot(cp)

#############################################################################
    ### experiments:1
    ### signals:40 subspaces:2 length:128
    ### noise sigma:0.05
    ### lambda1:0.01 lambda2:2 K:0
    n_signals=40
    K=0
    lambda1=0.01
    lambda2=2
    n_length=128
    n_experiments=1
    np.random.seed(0)
    cp_list=[]
    for i in range(n_experiments):
        select_y_list=[i for i in range(n_signals)]
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
        #plt.plot(cp)
        cp_temp=cp.tolist()
        cp_list.append(cp_temp)

    cp_matrix_1=np.array(cp_list)
    mean_cp=cp_matrix_1.mean(axis=0)
    std_cp=cp_matrix_1.std(axis=0)
    ### save LCP results
    #np.save('cp_matrix.npy',cp_matrix_1)
    
    ### plot LCP
    fig = plt.figure(figsize=(4.5,3))
    ax = fig.add_axes([0.1, 0.15, 0.85, 0.8]) 
    plt.errorbar(np.arange (1,129,1),mean_cp,yerr=std_cp,label='estimated location',fmt='o',ecolor='r',color='b',ms=2,elinewidth=1,capsize=1)
    plt.axvline(32,color='k',ls='-.',lw=1)
    plt.axvline(64,color='k',ls='-.',lw=1)
    plt.axhline(0,label='true location',color='k',ls='-.',lw=1)
    plt.axhline(32,color='k',ls='-.',lw=1)
    plt.axhline(64,color='k',ls='-.',lw=1)
    plt.xticks([0,32,64,96,128],fontsize=5)
    plt.yticks([0,16,32,48,64],fontsize=5)
    plt.xlim([0,129])
    plt.ylim([-16,80])
    plt.xlabel("Time",fontsize=8)
    plt.ylabel('The latest change point',fontsize=8)
    plt.legend(loc='upper left',fontsize=6)
    #plt.savefig('experiment1_LCP.png',dpi=300)
#############################################################################    
    #plot profiles and estimated coefs
    lambda1=0.01
    coef_matrix_large_t=np.zeros((3,n_signals,n_signals))
    for i in range(n_signals):
        cp_list=[0,128]
        cp_list=[0,32,64,128]
        coef_matrix=np.zeros((n_signals+3,len(cp_list)-1))
        data_y_estimate=data_noise[:,i].copy()
        for j in range(len(cp_list)-1):
            data_temp=data_noise[cp_list[j]:cp_list[j+1],:]
            data_y=data_temp[:,i].copy()
            data_x=data_temp.copy()
            data_x[:,i]=0
            lasso = Lasso(lambda1,fit_intercept=False)
            #lasso = LinearRegression()
            lasso.fit(data_x, data_y)
            coef=lasso.coef_
            r_square=lasso.score(data_x,data_y)
            data_y_estimate_temp=np.dot(data_x,coef)
            data_y_estimate[cp_list[j]:cp_list[j+1]]=data_y_estimate_temp
            cost=np.sum((data_y_estimate_temp-data_y)**2)+lambda1*np.linalg.norm(coef,ord=1)
            #var=np.var(data_y_estimate_temp-data_y)
            #print(var)
            coef_matrix[0:n_signals,j]=coef
            coef_matrix[n_signals,j]=0
            coef_matrix[n_signals+1,j]=cost
            coef_matrix[n_signals+2,j]=r_square
            coef_matrix_large_t[j,:,i]=coef_matrix[0:n_signals,j]
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.7]) 
        plt.plot(np.arange(1,129,1),data_noise[:,i],label='original',color='k')
        plt.plot(np.arange(1,129,1),data_y_estimate,label='fit',color='r',linestyle ='--')
        plt.title("Time series "+str(i+1),fontsize=20)
        plt.xticks([0,32,64,96,128],fontsize=20)
        plt.yticks(fontsize=20)
        #plt.savefig('./fig/128-n-10/Function '+str(i+1)+'.png')
        coef_matrix_large=np.zeros((8,n_signals))
        coef_matrix_large[0:2,:]=np.transpose(coef_matrix[0:n_signals,0])
        coef_matrix_large[2:4,:]=np.transpose(coef_matrix[0:n_signals,1])
        coef_matrix_large[4:8,:]=np.transpose(coef_matrix[0:n_signals,2])
        plt.figure()
        plt.imshow(np.transpose(coef_matrix_large),vmax=1,vmin=-1)
        #x = range(0,8,2)
        #plt.xticks(x,(0,32,64,96,128))
        #plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Function "+str(i+1))
        #plt.savefig('./fig/128-n-10/coef_matrix_large'+str(i+1)+'.png',dpi=300,bbox_inches='tight')
#############################################################################
    ###metrics
    cp_true=[32,64]
    delay_max=5
    ###offline: Precision, Recall
    Precision_list=[]
    Recall_list=[]
    for i in range(n_experiments):
        n_p_Precision=0
        n_p_Recall=0
        n_est=0
        cp_temp=cp_matrix_1[i,:]
        cp_return_list=cp_return(cp_temp)[1:]
        #Recall
        for j in range(len(cp_true)):
            cp_true_single=cp_true[j]
            for k in range(len(cp_return_list)):
                cp_est_single=cp_return_list[k]
                if abs(cp_true_single-cp_est_single)<=delay_max:
                    n_p_Recall+=1
                    break
        #Precision
        for k in range(len(cp_return_list)):
            cp_est_single=cp_return_list[k]
            for j in range(len(cp_true)):
                cp_true_single=cp_true[j]
                if abs(cp_true_single-cp_est_single)<=delay_max:
                    n_p_Precision+=1
                    break
        n_est+=len(cp_return_list)
        Precision_temp=n_p_Precision/n_est
        Recall_temp=n_p_Recall/len(cp_true)
        Precision_list.extend([Precision_temp])
        Recall_list.extend([Recall_temp])
    print("Precision: %.2f%%, %.2f%%" %(np.mean(Precision_list)*100,np.std(Precision_list)*100))
    print("Recall: %.2f%%, %.2f%%" %(np.mean(Recall_temp)*100,np.std(Recall_temp)*100))
    ###online: detection delay
    time_delay_list=[]
    for i in range(n_experiments):
        cp_temp=cp_matrix_1[i,:]  
        for j in range(len(cp_true)):
            cp_true_single=cp_true[j]
            fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
            time_delay=fastest_detection-cp_true_single
            time_delay_list.extend([time_delay])
    time_delay_mean=np.mean(time_delay_list)
    time_delay_std=np.std(time_delay_list)
    print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
#############################################################################
    ### compare with OP algorithm
    lambda1=0.01
    lambda2=2
    K=0
    n_signals=40
    select_y_list=[i for i in range(n_signals)]
    list_t_0_list=[]
    list_t_OP_list=[]
    np.random.seed(0)
    n_experiments=100
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        #cp_0,list_t_0,list_R_0=PELT_select_y_time(data_noise,lambda1,lambda2,0,select_y_list)
        #cp_K,list_t_K,list_R_K=PELT_select_y_time(data_noise,lambda1,lambda2,0.1,select_y_list)
        #cp_K_2,list_t_K_2,list_R_K_2=PELT_select_y_time(data_noise,lambda1,lambda2,0.2,select_y_list)
        #cp_K_3,list_t_K_3,list_R_K_3=PELT_select_y_time(data_noise,lambda1,lambda2,0.3,select_y_list)
        cp_OP,list_t_OP,list_R_K_OP=PELT_select_y_time(data_noise,lambda1,lambda2,-100,select_y_list)
        #list_t_0_list.append(list_t_0)
        list_t_OP_list.append(list_t_OP)

    lambda1=0.01
    lambda2=2
    K=0
    n_signals=40
    select_y_list=[i for i in range(n_signals)]
    np.random.seed(0)
    n_experiments=100
    list_t_02_list=[]
    cp_list_02=[]
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        cp_K_2,list_t_K_2,list_R_K_2=PELT_select_y_time(data_noise,lambda1,lambda2,0.2,select_y_list)
        list_t_02_list.append(list_t_K_2)
        cp_temp=cp_K_2.tolist()
        cp_list_02.append(cp_temp)
        
    time_total_mean=np.mean(np.array(list_t_02_list)[:,-1])
    time_total_std=np.std(np.array(list_t_02_list)[:,-1])
    print("Total time: mean=%.1f, std=%.1f" %(time_total_mean,time_total_std))
    time_per_mean=np.mean(np.diff(np.array(list_t_02_list)))
    time_per_std=np.mean(np.diff(np.array(list_t_02_list)))
    print("Computing time per step: mean=%.2f, std=%.2f" %(time_per_mean,time_per_std))
    ''' 
    time_total_mean=np.mean(np.array(list_t_0_list)[:,-1])
    time_total_std=np.std(np.array(list_t_0_list)[:,-1])
    print("Total time: mean=%.1f, std=%.1f" %(time_total_mean,time_total_std))
    time_per_mean=np.mean(np.diff(np.array(list_t_0_list)))
    time_per_std=np.mean(np.diff(np.array(list_t_0_list)))
    print("Computing time per step: mean=%.1f, std=%.1f" %(time_per_mean,time_per_std))
    '''
    time_total_mean_OP=np.mean(np.array(list_t_OP_list)[:,-1])
    print("Total time: OP=%.1f, PELT=%.1f" %(time_total_mean_OP,time_total_mean))

    np.save('list_t_02_list.npy',np.array(list_t_02_list))
    np.save('cp_list_02.npy',np.array(cp_list_02))
    
    np.save('list_t_OP_list.npy',np.array(list_t_OP_list))
    np.save('list_t_0_list.npy',np.array(list_t_0_list))
    
    plt.figure(figsize=(6,4))
    plt.plot(range(1,128),np.mean(np.diff(np.array(list_t_OP_list)),axis=0),label='OP',color='b',ls='-.',lw=2)
    plt.plot(range(1,128),np.max(np.diff(np.array(list_t_OP_list)),axis=0),color='b',ls='-.',lw=2,alpha=0.2)
    plt.plot(range(1,128),np.min(np.diff(np.array(list_t_OP_list)),axis=0),color='b',ls='-.',lw=2,alpha=0.2)
    plt.plot(range(1,128),np.mean(np.diff(np.array(list_t_02_list)),axis=0),label='PELT',color='black',ls='-',lw=2)
    plt.plot(range(1,128),np.max(np.diff(np.array(list_t_02_list)),axis=0),color='black',ls='-',lw=2,alpha=0.2)
    plt.plot(range(1,128),np.min(np.diff(np.array(list_t_02_list)),axis=0),color='black',ls='-',lw=2,alpha=0.2)
    plt.fill_between(range(1,128),np.min(np.diff(np.array(list_t_OP_list)),axis=0),np.max(np.diff(np.array(list_t_OP_list)),axis=0),color='b',alpha=0.2)
    plt.fill_between(range(1,128),np.min(np.diff(np.array(list_t_02_list)),axis=0),np.max(np.diff(np.array(list_t_02_list)),axis=0),color='black',alpha=0.2)
    #plt.plot(np.diff(np.array(list_t_K)),label='PELT K=0.1',color='r',ls='--',lw=2)
    #plt.plot(np.diff(np.array(list_t_K_2)),label='PELT K=0.2',color='purple',ls='-',lw=2)
    plt.axvline(32,color='k',ls='-.')
    plt.axvline(64,color='k',ls='-.')
    plt.xticks([0,32,64,96,128],fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Time",fontsize=10)
    plt.ylabel("Computation time per step (s)",fontsize=10)
    plt.legend()
    plt.savefig('time_per_step.png',dpi=300,bbox_inches='tight')

    array_R_K_2=np.zeros((129,129))
    for i in range(len(list_R_K_2)):
        list_temp=list_R_K_2[i]
        for j in range(len(list_temp)):
            array_R_K_2[128-list_temp[j],i+1]=1
    plt.figure(figsize=(12,4))
    plt.imshow(1-array_R_K_2,vmax=1,vmin=-1,cmap ='gray',label='candidates')
    plt.xlim([-1,127])
    #plt.ylim([-1,127])
    plt.xticks([0,32,64,96,128])
    plt.yticks([0,32,64,96,128],[128,96,64,32,0])
    #plt.colorbar()
    plt.xlabel('Time',fontsize=10)
    plt.ylabel('The candidates of LCP',fontsize=10)
    plt.plot(1+np.arange(0,32,1),np.ones(32)*128,color='gray',lw=1.5,label='candidates')
    plt.plot(1+np.arange(0,32,1),np.ones(32)*128,color='r',linestyle=(0,(3,2)),lw=1.5,label='true LCP')
    plt.plot(33+np.arange(0,32,1),np.ones(32)*128-33,color='r',linestyle=(0,(3,2)),lw=1.5)
    plt.plot(65+np.arange(0,64,1),np.ones(64)*128-65,color='r',linestyle=(0,(3,2)),lw=1.5)
    plt.axvline(96,color='b',ls='--')
    ax = plt.gca()#获取边框
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    plt.legend(loc='upper left')
    plt.savefig('LCP_candidates_K1.png',dpi=300,bbox_inches='tight')

#############################################################################
    #compare with DFSL
    n_signals=40
    n_length=128
    n_experiments=100
    np.random.seed(0)
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch.mat', {'data_patch': data_patch})
    sio.savemat('data_patch_clean.mat', {'data_patch_clean': data_patch_clean})
    
#############################################################################
    #high-dimensional
    num_data=1
    n_signals=400
    n_length=320
    data_past=np.zeros((num_data,64,n_signals))
    np.random.seed(0)
    for j in range(num_data):
        data=simulation_large(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_past[j,:,:]=data_noise[:64,:]
        
    cp_list=[i*32 for i in range(3)]
    K=0
    lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K)
    print(lambda1,lambda2)
    
    n_signals=400
    K=2
    lambda1=0.01
    lambda2=5
    n_length=320
    n_experiments=100
    np.random.seed(0)
    cp_list=[]
    list_t_list=[]
    for i in range(n_experiments):
        print(i)
        select_y_list=[i for i in range(n_signals)]
        data=simulation_large(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        cp,list_t,list_R=PELT_select_y_time(data_noise,lambda1,lambda2,K,select_y_list)
        list_t_list.append(list_t)
        cp_temp=cp.tolist()
        cp_list.append(cp_temp)

    cp_matrix_1=np.array(cp_list)
    mean_cp=cp_matrix_1.mean(axis=0)
    std_cp=cp_matrix_1.std(axis=0)
    ### save LCP results
    #np.save('cp_matrix.npy',cp_matrix_1)
    
    cp_true=[(i+1)*32 for i in range(9)]
    delay_max=5
    ###offline: Precision, Recall
    n_p_Precision=0
    n_p_Recall=0
    n_est=0
    for i in range(n_experiments):
        cp_temp=cp_matrix_1[i,:]
        cp_return_list=cp_return(cp_temp)[1:]
        print(cp_return_list)
        #Recall
        for j in range(len(cp_true)):
            cp_true_single=cp_true[j]
            for k in range(len(cp_return_list)):
                cp_est_single=cp_return_list[k]
                if abs(cp_true_single-cp_est_single)<=delay_max:
                    n_p_Recall+=1
                    break
        #Precision
        for k in range(len(cp_return_list)):
            cp_est_single=cp_return_list[k]
            for j in range(len(cp_true)):
                cp_true_single=cp_true[j]
                if abs(cp_true_single-cp_est_single)<=delay_max:
                    n_p_Precision+=1
                    break
        n_est+=len(cp_return_list)
    Precision=n_p_Precision/n_est
    Recall=n_p_Recall/len(cp_true)/n_experiments
    print("Precision=%.2f%%, Recall=%.2f%%" %(Precision*100,Recall*100))
    ###online: detection delay
    time_delay_list=[]
    for i in range(n_experiments):
        cp_temp=cp_matrix_1[i,:]  
        for j in range(len(cp_true)):
            cp_true_single=cp_true[j]
            if np.shape(np.where(abs(cp_temp-cp_true_single)<=delay_max))[1]>0:
                fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
                time_delay=fastest_detection-cp_true_single
                time_delay_list.extend([time_delay])
    time_delay_mean=np.mean(time_delay_list)
    time_delay_std=np.std(time_delay_list)
    print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
    
    time_total_mean=np.mean(np.array(list_t_list)[:,-1])
    time_total_std=np.std(np.array(list_t_list)[:,-1])
    print("Total time: mean=%.1f, std=%.1f" %(time_total_mean,time_total_std))
    time_per_mean=np.mean(np.diff(np.array(list_t_list)))
    time_per_std=np.mean(np.diff(np.array(list_t_list)))
    print("Computing time per step: mean=%.1f, std=%.1f" %(time_per_mean,time_per_std))

#############################################################################
    #compare with DFSL
    n_signals=400
    n_length=320
    n_experiments=100
    np.random.seed(0)
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    for i in range(n_experiments):
        data=simulation_large(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_large.mat', {'data_patch': data_patch})
    sio.savemat('data_patch_clean_large.mat', {'data_patch_clean': data_patch_clean})
