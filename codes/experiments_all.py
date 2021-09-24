from functions import *
#from tunning import *
import numpy as np                  
import scipy.io as sio
import time

if __name__ == '__main__':
#%%
    ### experiments:1
    ### signals:40 subspaces:2 length:128
    ### noise sigma:0.05,0.1,0.2
    
    num_data=3
    n_signals=40
    n_length=128
    sigma2_list=[0.05,0.1,0.2]

    data_past=np.zeros((num_data,n_length,n_signals))
    results_list_1=[]
    index_experiment=-1
    for sigma2 in sigma2_list:
        np.random.seed(7)
        for j in range(num_data):
            data=simulation(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_past[j,:,:]=data_noise
            
        cp_list=[0,32,64,128]
        K=0
        lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K) 
        index_experiment+=1
        n_signals=40
        K=lambda2/1.5
        n_length=128
        n_experiments=100
        np.random.seed(0)
        cp_list=[]
            
        data_patch=np.zeros((n_experiments,n_length,n_signals))
        data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
        np.random.seed(0)
        for i in range(n_experiments):
            data=simulation(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_patch[i,:,:]=data_noise
            data_patch_clean[i,:,:]=data
        
        time_list=[]
        for i in range(n_experiments):
            select_y_list=[i for i in range(n_signals)]
            data=data_patch_clean[i,:,:]
            data_noise=data_patch[i,:,:]
            time_start=time.time()
            cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
            time_running=time.time()-time_start
            time_list.extend([time_running])
            #plt.plot(cp)
            cp_temp=cp.tolist()
            cp_list.append(cp_temp)
            print('Sigma=%.2f, %d-th experiment'%(sigma2,i+1))

        mean_time_running=np.mean(time_list)
        std_time_running=np.std(time_list)
        print("Total Time: %.2f, %.2f" %(mean_time_running,std_time_running))
        cp_matrix_1=np.array(cp_list)
        np.save("results_list_1_"+str(sigma2)+'.npy',cp_matrix_1)

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
            if n_est>0:
                Precision_temp=n_p_Precision/n_est
                Precision_list.extend([Precision_temp])
            Recall_temp=n_p_Recall/len(cp_true)
            Recall_list.extend([Recall_temp])
        precision_mean=np.mean(Precision_list)*100
        precision_std=np.std(Precision_list)*100
        recall_mean=np.mean(Recall_list)*100
        recall_std=np.std(Recall_list)*100
        print("Precision: %.2f%%, %.2f%%" %(precision_mean,precision_std))
        print("Recall: %.2f%%, %.2f%%" %(recall_mean,recall_std))
        ###online: detection delay
        time_delay_list=[]
        for i in range(n_experiments):
            cp_temp=cp_matrix_1[i,:]  
            for j in range(len(cp_true)):
                cp_true_single=cp_true[j]
                find_fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0]
                if len(find_fastest_detection)>0:
                    fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
                    time_delay=fastest_detection-cp_true_single
                    time_delay_list.extend([time_delay])
        time_delay_mean=np.mean(time_delay_list)
        time_delay_std=np.std(time_delay_list)
        print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
        results=[sigma2,lambda1,lambda2,K,mean_time_running,std_time_running,precision_mean,precision_std,recall_mean,recall_std,time_delay_mean,time_delay_std]
        results_list_1.append(results)
    np.save("results_list_1.npy",results_list_1)

#%%
    '''
    #saved as .mat
    n_signals=40
    n_length=128
    n_experiments=100
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_1.mat', {'data_patch': data_patch})
    sio.savemat('data_patch_clean.mat', {'data_patch_clean': data_patch_clean})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.1
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_2.mat', {'data_patch': data_patch})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.2
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_3.mat', {'data_patch': data_patch})
    '''
#%%
    ### experiments:2
    ### signals:40 subspaces:2 length:128
    ### noise sigma:0.05,0.1,0.2
    
    num_data=3
    n_signals=40
    n_length=128
    sigma2_list=[0.05,0.1,0.2]
    data_past=np.zeros((num_data,n_length,n_signals))
    results_list_2=[]
    for sigma2 in sigma2_list:
        np.random.seed(7)
        for j in range(num_data):
            data=simulation_similar(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_past[j,:,:]=data_noise
            
        cp_list=[0,32,64,128]
        K=0
        lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K) 
        
        n_signals=40
        K=lambda2/1.5
        n_length=128
        n_experiments=100
        np.random.seed(0)
        cp_list=[]
            
        data_patch=np.zeros((n_experiments,n_length,n_signals))
        data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
        np.random.seed(0)
        for i in range(n_experiments):
            data=simulation_similar(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_patch[i,:,:]=data_noise
            data_patch_clean[i,:,:]=data

        time_list=[]
        for i in range(n_experiments):
            select_y_list=[i for i in range(n_signals)]
            data=data_patch_clean[i,:,:]
            data_noise=data_patch[i,:,:]
            time_start=time.time()
            cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
            time_running=time.time()-time_start
            time_list.extend([time_running])
            #plt.plot(cp)
            cp_temp=cp.tolist()
            cp_list.append(cp_temp)
            print('Sigma=%.2f, %d-th experiment'%(sigma2,i+1))

        mean_time_running=np.mean(time_list)
        std_time_running=np.std(time_list)
        print("Total Time: %.2f, %.2f" %(mean_time_running,std_time_running))
        cp_matrix_1=np.array(cp_list)
        np.save("results_list_2_"+str(sigma2)+'.npy',cp_matrix_1)

            
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
            if n_est>0:
                Precision_temp=n_p_Precision/n_est
                Precision_list.extend([Precision_temp])
            Recall_temp=n_p_Recall/len(cp_true)
            Recall_list.extend([Recall_temp])
        precision_mean=np.mean(Precision_list)*100
        precision_std=np.std(Precision_list)*100
        recall_mean=np.mean(Recall_list)*100
        recall_std=np.std(Recall_list)*100
        print("Precision: %.2f%%, %.2f%%" %(precision_mean,precision_std))
        print("Recall: %.2f%%, %.2f%%" %(recall_mean,recall_std))
        ###online: detection delay
        time_delay_list=[]
        for i in range(n_experiments):
            cp_temp=cp_matrix_1[i,:]  
            for j in range(len(cp_true)):
                cp_true_single=cp_true[j]
                find_fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0]
                if len(find_fastest_detection)>0:
                    fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
                    time_delay=fastest_detection-cp_true_single
                    time_delay_list.extend([time_delay])
        time_delay_mean=np.mean(time_delay_list)
        time_delay_std=np.std(time_delay_list)
        print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
        results=[sigma2,lambda1,lambda2,K,mean_time_running,std_time_running,precision_mean,precision_std,recall_mean,recall_std,time_delay_mean,time_delay_std]
        results_list_2.append(results)
    np.save("results_list_2.npy",results_list_2)

#%%
    '''
    n_signals=40
    n_length=128
    n_experiments=100
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_1.mat', {'data_patch': data_patch})
    sio.savemat('data_patch_clean.mat', {'data_patch_clean': data_patch_clean})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.1
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_2.mat', {'data_patch': data_patch})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.2
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_3.mat', {'data_patch': data_patch})
    '''
#%%
    ### experiments:3
    ### signals:400 subspaces:2 length:320
    ### noise sigma:0.05,0.1,0.2
    
    num_data=1
    n_signals=400
    n_length=320
    n_length_small=128
    sigma2_list=[0.05,0.1,0.2]
    lambda1_list=[0.05,0.05,0.05]
    lambda2_list=[100,120,150]
    
    data_past=np.zeros((num_data,n_length_small,n_signals))
    results_list_3=[]
    for sigma2_index in range(3):
        sigma2=sigma2_list[sigma2_index]
        print(sigma2)
        '''
        np.random.seed(7)
        for j in range(num_data):
            data=simulation(n_length_small,n_signals)
            data_noise=data+np.random.normal(size=(n_length_small,n_signals))*sigma2
            data_past[j,:,:]=data_noise
            
        #cp_list=[i*32 for i in range(11)]
        cp_list=[0,32,64,128]
        K=0
        lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K) 
        '''
        n_signals=400
        lambda1=lambda1_list[sigma2_index]
        lambda2=lambda2_list[sigma2_index]
        K=lambda2/1.2
        n_length=320
        n_experiments=10
        np.random.seed(0)
        cp_list=[]
            
        data_patch=np.zeros((n_experiments,n_length,n_signals))
        data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
        np.random.seed(0)
        for i in range(n_experiments):
            data=simulation_large(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_patch[i,:,:]=data_noise
            data_patch_clean[i,:,:]=data
        
        time_list=[]
        for i in range(n_experiments):
            select_y_list=[i for i in range(n_signals)]
            data=data_patch_clean[i,:,:]
            data_noise=data_patch[i,:,:]
            time_start=time.time()
            cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
            time_running=time.time()-time_start
            time_list.extend([time_running])
            #plt.plot(cp)
            cp_temp=cp.tolist()
            cp_list.append(cp_temp)
            print('Sigma=%.2f, %d-th experiment'%(sigma2,i+1))
        mean_time_running=np.mean(time_list)
        std_time_running=np.std(time_list)
        print("Total Time: %.2f, %.2f" %(mean_time_running,std_time_running))
        cp_matrix_1=np.array(cp_list)
        np.save("results_list_3_"+str(sigma2)+'.npy',cp_matrix_1)
            
        cp_true=[i*32 for i in range(1,10)]
        delay_max=6
        ###offline: Precision, Recall
        Precision_list=[]
        Recall_list=[]
        for i in range(len(cp_matrix_1)):
            n_p_Precision=0
            n_p_Recall=0
            n_est=0
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
            if n_est>0:
                Precision_temp=n_p_Precision/n_est
                Precision_list.extend([Precision_temp])
            Recall_temp=n_p_Recall/len(cp_true)
            Recall_list.extend([Recall_temp])
        precision_mean=np.mean(Precision_list)*100
        precision_std=np.std(Precision_list)*100
        recall_mean=np.mean(Recall_list)*100
        recall_std=np.std(Recall_list)*100
        print("Precision: %.2f%%, %.2f%%" %(precision_mean,precision_std))
        print("Recall: %.2f%%, %.2f%%" %(recall_mean,recall_std))
        ###online: detection delay
        time_delay_list=[]
        for i in range(len(cp_matrix_1)):
            cp_temp=cp_matrix_1[i,:]  
            for j in range(len(cp_true)):
                cp_true_single=cp_true[j]
                find_fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0]
                if len(find_fastest_detection)>0:
                    fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
                    time_delay=fastest_detection-cp_true_single
                    time_delay_list.extend([time_delay])
        time_delay_mean=np.mean(time_delay_list)
        time_delay_std=np.std(time_delay_list)
        print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
        results=[sigma2,lambda1,lambda2,K,mean_time_running,std_time_running,precision_mean,precision_std,recall_mean,recall_std,time_delay_mean,time_delay_std]
        results_list_3.append(results)
    np.save("results_list_3.npy",results_list_3)

#%%
    '''
    n_signals=40
    n_length=128
    n_experiments=100
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_1.mat', {'data_patch': data_patch})
    sio.savemat('data_patch_clean.mat', {'data_patch_clean': data_patch_clean})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.1
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_2.mat', {'data_patch': data_patch})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.2
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_3.mat', {'data_patch': data_patch})
    '''
#%%
    ### experiments:4
    ### signals:40 subspaces:2 length:128
    ### noise sigma:0.05,0.1,0.2
    
    num_data=3
    n_signals=40
    n_length=128
    sigma2_list=[0.05,0.1,0.2]
    data_past=np.zeros((num_data,n_length,n_signals))
    results_list_4=[]
    for sigma2 in sigma2_list:
        np.random.seed(7)
        for j in range(num_data):
            data=simulation_graph(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_past[j,:,:]=data_noise
            
        cp_list=[0,32,64,128]
        K=0
        lambda1,lambda2=determine_parameters(data_past,n_signals,cp_list,K) 
        
        n_signals=40
        K=lambda2/1.5
        n_length=128
        n_experiments=100
        np.random.seed(0)
        cp_list=[]
            
        data_patch=np.zeros((n_experiments,n_length,n_signals))
        data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
        np.random.seed(0)
        for i in range(n_experiments):
            data=simulation_graph(n_length,n_signals)
            data_noise=data+np.random.normal(size=(n_length,n_signals))*sigma2
            data_patch[i,:,:]=data_noise
            data_patch_clean[i,:,:]=data
        
        time_list=[]
        for i in range(n_experiments):
            select_y_list=[i for i in range(n_signals)]
            data=data_patch_clean[i,:,:]
            data_noise=data_patch[i,:,:]
            time_start=time.time()
            cp=PELT_select_y(data_noise,lambda1,lambda2,K,select_y_list)
            time_running=time.time()-time_start
            time_list.extend([time_running])
            #plt.plot(cp)
            cp_temp=cp.tolist()
            cp_list.append(cp_temp)
            print('Sigma=%.2f, %d-th experiment'%(sigma2,i+1))
        mean_time_running=np.mean(time_list)
        std_time_running=np.std(time_list)
        print("Total Time: %.2f, %.2f" %(mean_time_running,std_time_running))
        cp_matrix_1=np.array(cp_list)
        np.save("results_list_4_"+str(sigma2)+'.npy',cp_matrix_1)
            
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
            if n_est>0:
                Precision_temp=n_p_Precision/n_est
                Precision_list.extend([Precision_temp])
            Recall_temp=n_p_Recall/len(cp_true)
            Recall_list.extend([Recall_temp])
        precision_mean=np.mean(Precision_list)*100
        precision_std=np.std(Precision_list)*100
        recall_mean=np.mean(Recall_list)*100
        recall_std=np.std(Recall_list)*100
        print("Precision: %.2f%%, %.2f%%" %(precision_mean,precision_std))
        print("Recall: %.2f%%, %.2f%%" %(recall_mean,recall_std))
        ###online: detection delay
        time_delay_list=[]
        for i in range(n_experiments):
            cp_temp=cp_matrix_1[i,:]  
            for j in range(len(cp_true)):
                cp_true_single=cp_true[j]
                find_fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0]
                if len(find_fastest_detection)>0:
                    fastest_detection=np.where(abs(cp_temp-cp_true_single)<=delay_max)[0].min()
                    time_delay=fastest_detection-cp_true_single
                    time_delay_list.extend([time_delay])
        time_delay_mean=np.mean(time_delay_list)
        time_delay_std=np.std(time_delay_list)
        print("Time delay: mean=%.1f, std=%.1f" %(time_delay_mean,time_delay_std))
        results=[sigma2,lambda1,lambda2,K,mean_time_running,std_time_running,precision_mean,precision_std,recall_mean,recall_std,time_delay_mean,time_delay_std]
        results_list_4.append(results)
    np.save("results_list_4.npy",results_list_4)
#%%
    '''
    n_signals=40
    n_length=128
    n_experiments=100
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.05
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_1.mat', {'data_patch': data_patch})
    sio.savemat('data_patch_clean.mat', {'data_patch_clean': data_patch_clean})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.1
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_2.mat', {'data_patch': data_patch})
    
    data_patch=np.zeros((n_experiments,n_length,n_signals))
    data_patch_clean=np.zeros((n_experiments,n_length,n_signals))
    np.random.seed(0)
    for i in range(n_experiments):
        data=simulation(n_length,n_signals)
        data_noise=data+np.random.normal(size=(n_length,n_signals))*.2
        data_patch[i,:,:]=data_noise
        data_patch_clean[i,:,:]=data
    sio.savemat('data_patch_3.mat', {'data_patch': data_patch})
    '''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
