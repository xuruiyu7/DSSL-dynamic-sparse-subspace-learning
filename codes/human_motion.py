from functions import *
from tunning import *
import seaborn as sns
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import scipy.io as sio



if __name__ == "__main__":    
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    ###load data
    data_path_bow=r'../GestureDataset/P1_2_7_p16.csv'
    data_path_shoot=r'../GestureDataset/P1_2_6_p16.csv'
    data_path_throw=r'../GestureDataset/P1_2_8_p16.csv'
    data_bow=load_data(data_path_bow)
    data_shoot=load_data(data_path_shoot)
    data_throw=load_data(data_path_throw)
#%%
    shoot_cp=[168,327,486,637,774,902,1045,1211,1394,1572]
    throw_cp=[90,206,313,419,530,636,738,855,965,1069]
        
    cp_list_result=[]
    data_combine_large=np.zeros((10,210,18))
    for ii in range(10):
        data_shoot_once=data_shoot[shoot_cp[ii]-120:shoot_cp[ii],:]
        data_throw_once=data_throw[throw_cp[ii]:throw_cp[ii]+90,:]
        diff=data_shoot_once[-1,:]-data_throw_once[0,:]
        diff_x=diff[0::3].mean()
        diff_z=diff[1::3].mean()
        diff_y=diff[2::3].mean()
        diff_list=[diff_x,diff_z,diff_y]
        for i in range(len(data_throw_once)):
            for j in range(60):
                data_throw_once[i,j]=data_throw_once[i,j]+diff[j]
        data_combine=np.vstack((data_shoot_once,data_throw_once))
    
        data=data_combine
        data_smooth=data_combine.copy()
        b, a = signal.butter(8, 0.2)
        for i in range(60):
            filtedData = signal.filtfilt(b, a, data[:,i])
            data_smooth[:,i]=filtedData
        data_r_square=np.zeros((210,20))
        data_r=np.zeros((210,20))
        for i in range(210):
            for j in range(20):
                data_r_square[i,j]=data_smooth[i,3*j]**2+data_smooth[i,3*j+1]**2+data_smooth[i,3*j+2]**2
                data_r[i,j]=np.sqrt(data_r_square[i,j])

        data_r_delete=np.delete(data_r, [6,10], axis=1)
        data_combine_large[ii,:,:]=data_r_delete
        #sio.savemat('data_combine_large.mat', {'data_combine_large': data_combine_large})

    ###PELT
    cp_list=[0,120,210]
    n_signals=18
    data_past=data_combine_large[0:1,:,:]
    K=0
    lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K,delta=10)
    
    #lambda1=0.0068
    #lambda2=0.35
    cp_list_result=[]
    for ii in range(10):
        data_r_delete=data_combine_large[ii,:,:]
        select_y_list=[i for i in range(18)]
        cp=PELT_select_y(data_r_delete,lambda1,lambda2,K,select_y_list)
        cp_list_result.append(cp.tolist())

    cp_matrix_1=np.array(cp_list_result)
    mean_cp=cp_matrix_1.mean(axis=0)
    std_cp=cp_matrix_1.std(axis=0)
    
    '''
    for i in range(10):
        cp_list_temp=cp_return(cp_matrix_1[i,:])
        print(cp_list_temp)
    '''
    '''
    fig = plt.figure(figsize=(4.5,3))
    ax = fig.add_axes([0.1, 0.15, 0.85, 0.8]) 
    plt.errorbar(np.arange (1,211,1),mean_cp,yerr=std_cp,label='estimated location',fmt='o',ecolor='r',color='b',ms=2,elinewidth=1,capsize=1)
    plt.axvline(120,color='k',ls='-.',lw=1)
    plt.axhline(0,label='true location',color='k',ls='-.',lw=1)
    plt.axhline(120,color='k',ls='-.',lw=1)
    plt.xlabel("Time",fontsize=8)
    plt.ylabel('The latest change point',fontsize=8)
    plt.legend(loc='upper left',fontsize=6)
    '''
    
    list_t_0_list=[]
    cp_list_result=[]
    for ii in range(10):
        data_r_delete=data_combine_large[ii,:,:]
        select_y_list=[i for i in range(18)]
        cp_0,list_t_0,list_R_0=PELT_select_y_time(data_r_delete,lambda1,lambda2,K,select_y_list)
        cp_list_result.append(cp_0)
        list_t_0_list.append(list_t_0)

    time_total_mean=np.mean(np.array(list_t_0_list)[:,-1])
    time_total_std=np.std(np.array(list_t_0_list)[:,-1])
    print("Total time: mean=%.1f, std=%.1f" %(time_total_mean,time_total_std))
    time_per_mean=np.mean(np.diff(np.array(list_t_0_list)))
    time_per_std=np.mean(np.diff(np.array(list_t_0_list)))
    print("Computing time per step: mean=%.2f, std=%.2f" %(time_per_mean,time_per_std))

    cp_matrix_1=np.array(cp_list_result)
    mean_cp=cp_matrix_1.mean(axis=0)
    std_cp=cp_matrix_1.std(axis=0)
    
    cp_true=[120]
    delay_max=10
    
    ###offline: Precision, Recall
    n_p_Precision=0
    n_p_Recall=0
    n_est=0
    for i in range(10):
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
    Precision=n_p_Precision/n_est
    Recall=n_p_Recall/len(cp_true)/10
    print("Precision=%.2f%%, Recall=%.2f%%" %(Precision*100,Recall*100))    
    
    time_delay_list=[]
    for i in [0,1,2,3,4,5,6,7,8]:
        cp_temp=cp_matrix_1[i,:]  
        for j in range(len(cp_true)):
            cp_true_single=cp_true[j]
            fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
            time_delay=fastest_detection-cp_true_single
            time_delay_list.extend([time_delay])
    time_delay_mean=np.mean(time_delay_list)
    time_delay_std=np.std(time_delay_list)
    print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
        
    data=data_combine_large[0,:,:]
    cp_list=[0,120,210]
    for i in range(18):
        coef_matrix=np.zeros((21,len(cp_list)-1))
        data_y_estimate=data[:,i].copy()
        for j in range(len(cp_list)-1):
            data_temp=data[cp_list[j]:cp_list[j+1],:]
            data_y=data_temp[:,i].copy()
            data_x=data_temp.copy()
            data_x[:,i]=0
            lasso = Lasso(lambda1,fit_intercept=True)
            lasso.fit(data_x, data_y)
            coef=lasso.coef_
            intercept=lasso.intercept_
            r_square=lasso.score(data_x,data_y)
            data_y_estimate_temp=np.dot(data_x,coef)+intercept
            data_y_estimate[cp_list[j]:cp_list[j+1]]=data_y_estimate_temp
            cost=np.sum((data_y_estimate_temp-data_y)**2)+lambda1*np.linalg.norm(coef,ord=1)
            coef_matrix[0:18,j]=coef
            coef_matrix[18,j]=intercept
            coef_matrix[19,j]=cost
            coef_matrix[20,j]=r_square
        
        fig=plt.figure(figsize=(20,1))
        ax = fig.add_axes([0, 0, 1, 1]) 
        plt.axis('off')
        plt.xlim((0,210))
        #plt.ylim((data_y_estimate.mean()-0.2,data_y_estimate.mean()+0.2))
        plt.plot(data[:,i],label='original',color='k',lw=1)
        plt.plot(data_y_estimate,label='fitted',color='r',linestyle=(2,(5.5,4.0)),lw=2)
        plt.axvline(120,color='black',linestyle='-.')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(data_y_estimate.mean()-0.55,data_y_estimate.mean()+0.5)
        plt.savefig('./plot/human_fitted/human_fitted_'+str(i+1)+'.png',dpi=300)
        
        len_x=11
        coef_matrix_large=np.zeros((len_x,18))
        coef_matrix_large[0:int(len_x/21*12),:]=np.transpose(coef_matrix[0:18,0])
        coef_matrix_large[int(len_x/21*12):len_x,:]=np.transpose(coef_matrix[0:18,1])
        plt.figure(figsize=(14/7,40/7))
        plt.imshow(np.transpose(coef_matrix_large),vmax=1,vmin=-1)
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
        plt.xlabel('Time')
        #plt.savefig('./fig/gesture/coef_matrix_large'+str(i+1)+'.png',dpi=300,bbox_inches='tight')
    
    fig=plt.figure(figsize=(23,2))
    ax = fig.add_axes([0.05, 0.2, 0.93, 0.75]) 
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.plot(true_cp_dis,label='true',color='k')
    plt.plot(range(1,211),cp_matrix_1[0],label='estimated',color='b',linestyle='-')
    #plt.scatter(np.array(range(210))+1,cp,label='estimated',color='b',s=10)
    plt.axvline(120,color='black',linestyle='-.')
    plt.axhline(120,color='black',linestyle='-.')
    #plt.ylabel('distance',fontsize=20)
    plt.xlim((0,211))
    plt.xticks([0,60,120,130,180,210],fontsize=20)
    plt.yticks([0,60,120],fontsize=20)
    plt.savefig('./plot/human_fitted/lcp.png',dpi=300)

    data=data_combine_large[0,:,:]
    cp=PELT_select_y(data,lambda1,lambda2,K,select_y_list)
    cp_list=cp_return(cp)
    cp_list.extend([210])
    for j in range(len(cp_list)-1):
        coef_matrix=np.zeros((18,18))
        for i in range(18):
            data_y_estimate=data[:,i].copy()
            data_temp=data[cp_list[j]:cp_list[j+1],:]
            data_y=data_temp[:,i].copy()
            data_x=data_temp.copy()
            data_x[:,i]=0
            lasso = Lasso(lambda1,fit_intercept=True)
            lasso.fit(data_x, data_y)
            coef=lasso.coef_
            intercept=lasso.intercept_
            r_square=lasso.score(data_x,data_y)
            data_y_estimate_temp=np.dot(data_x,coef)+intercept
            data_y_estimate[cp_list[j]:cp_list[j+1]]=data_y_estimate_temp
            cost=np.sum((data_y_estimate_temp-data_y)**2)+lambda1*np.linalg.norm(coef,ord=1)
            coef_matrix[0:18,i]=coef
        print(np.where((coef_matrix>0.3)|(coef_matrix<-0.3)))
        plt.figure()
        plt.imshow(np.transpose(coef_matrix),vmax=1,vmin=-1)
        plt.colorbar()
