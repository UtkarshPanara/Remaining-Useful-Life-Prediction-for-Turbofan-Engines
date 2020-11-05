# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:07:24 2020

@author: Utkarsh Panara
"""

#Import Libraries
import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic = True 

seed = 25

def set_seed(seed):
    torch.manual_seed(25)
    torch.cuda.manual_seed(25)
    np.random.seed(25)
    return None

def load_dataset(read_path, sub_dataset, max_life, cluster, train, test, finaltest):
    print("Loading Subdata set: ", sub_dataset)
    if train or test:
        if clusters:
            data_set = pd.read_csv(read_path + "train_FD"+str(sub_dataset)+'_cluster.csv')
        else:
            data_set = pd.read_csv(read_path + "train_FD"+str(sub_dataset)+'.csv')
            
    elif finaltest:
        if cluster:
            data_set = pd.read_csv(read_path + "test_FD"+str(sub_dataset)+'_cluster.csv')
        else:
            data_set = pd.read_csv(read_path + "test_FD"+str(sub_dataset)+'.csv')
    
    else:
        print("====Please check the argument Train/Test/FinalTest====")

    if train:
        data_set['RUL'] = compute_rul(read_path, sub_dataset, data_set, max_life, finaltest=False)
        print('\n\t Train FD ' + sub_dataset +'input shape:', data_set.shape)

    elif test:
        val=[]
        unique_id = pd.unique(data_set['engine_id']) 
        unique_20 = unique_id[-int(0.2*len(unique_id)):]
        data_gp = data_set.groupby('engine_id',sort = False)
        
        for engine_id, data in data_gp:
            if engine_id in unique_20:
                val.append(data)
        data_set = pd.concat(val, ignore_index=True, sort=False)

        data_set['RUL'] = compute_rul(read_path, sub_dataset,data_set, max_life, finaltest=False)
        print('\n\t Validatiom FD' + sub_dataset +'input shape:' , data_set.shape)

    elif finaltest:
        data_set['RUL'] = compute_rul(read_path, sub_dataset, data_set, max_life, finaltest=True)

        print('\n\t FinalTest FD' + sub_dataset +'input shape:' , data_set.shape)

    return data_set
    
def knee_RUL(cycle_list, max_cycle, MAXLIFE):
    '''
    Piecewise linear function with zero gradient and unit gradient
            ^
            |
    MAXLIFE |-----------
            |            \
            |             \
            |              \
            |               \
            |                \
            |----------------------->
    '''
    knee_RUL = []
    if max_cycle >= MAXLIFE:
        knee_point = max_cycle - MAXLIFE
        
        for i in range(0, len(cycle_list)):
            if i < knee_point:
                knee_RUL.append(MAXLIFE)
            else:
                tmp = knee_RUL[i - 1] - (MAXLIFE / (max_cycle - knee_point))
                knee_RUL.append(tmp)
    else:
        knee_point = MAXLIFE
        print("=========== knee_point < MAXLIFE ===========")
        for i in range(0, len(cycle_list)):
            knee_point -= 1
            knee_RUL.append(knee_point)
            
    return knee_RUL

def compute_rul(read_path, sub_dataset, data_set, max_life, finaltest=False, id='engine_id'):
    MAXLIFE = max_life
    rul = []
    true_rul = pd.read_csv(read_path + "RUL_FD"+sub_dataset + ".csv")
    unique_set = pd.unique(data_set['engine_id'])
    for _id in unique_set :
        unique_engine_ID = data_set[data_set[id] == _id]
        cycle_list = unique_engine_ID['cycle'].tolist()
        
        if finaltest:
            max_cycle = max(cycle_list) + true_rul['RUL'][_id-1]
        else:
            max_cycle = max(cycle_list)
        if max_cycle < MAXLIFE:
            print('<130 _id {} | max_cycle {}'.format(_id, max_cycle))
        rul.extend(knee_RUL(cycle_list, max_cycle, MAXLIFE))
    
    print('Lenght of input data RUL' , len(rul))
        
    return rul
                                      
def get_scaleing(data, scaler, scaler_range, train):
    sensors=['s1', 's2', 's3','s4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                       's15', 's16', 's17', 's18', 's19', 's20', 's21']
    operating_condition=["oc_0","oc_1","oc_2","oc_3","oc_4","oc_5"]

    global scalingparams

    if "operating_condition" in data.columns:
        print("\n\t Scaling by Clusters")
        groupby_oc = data.groupby("operating_condition", sort=False)
        scaled_sensors=[]
    
        if train:
            scalingparams = {}
            scalingparams['scaler'] = scaler
            scalingparams['scaler_range'] = scaler_range
        
            for operating_condition, data in groupby_oc:
                
                if scalingparams['scaler'] == 'mm':    
                    min_=np.min(data[sensors])
                    max_ =np.max(data[sensors])
                    
                    scaled_data=(((data[sensors]-min_)/(max_- min_))*(scalingparams['scaler_range'][1]-scalingparams['scaler_range'][0]))+scalingparams['scaler_range'][0]
    
                    scalingparams['min_oc'+str(operating_condition)]=min_
                    scalingparams['max_oc'+str(operating_condition)]=max_
                elif scalingparams['scaler'] == 'ss':
                    mean_=np.mean(data[sensors])
                    std_=np.std(data[sensors])
    
                    scaled_data=(data[sensors]-mean_)/std_
    
                    scalingparams['mean_oc'+str(operating_condition)]=mean_
                    scalingparams['std_oc'+str(operating_condition)]=std_
                scaled_sensors.append(scaled_data)
        else:
            for operating_condition, data in groupby_oc:
    
                if scalingparams['scaler'] == 'mm':    
                    min_=scalingparams['min_oc'+str(operating_condition)]
                    max_=scalingparams['max_oc'+str(operating_condition)]
                    scalingparams['scaler_range'] = scaler_range

                    scaled_data=(((data[sensors]-min_)/(max_- min_))*(scalingparams['scaler_range'][1]-scalingparams['scaler_range'][0]))+scalingparams['scaler_range'][0]
    
                elif scalingparams['scaler'] == 'ss':
                    mean_=scalingparams['mean_oc'+str(operating_condition)]
                    std_=scalingparams['std_oc'+str(operating_condition)]
                    
                    scaled_data=(data[sensors]-mean_)/std_
    
                scaled_sensors.append(scaled_data)
                
        scaled_df = pd.concat(scaled_sensors, sort=False)
        scaled_df=scaled_df.sort_index(axis=0, ascending=True)

    else:    
        print("\n\t Scaling Without Clusters")
        if train:
            scalingparams = {}
            scalingparams['scaler'] = scaler
            scalingparams['scaler_range'] = scaler_range
                        
            if scalingparams['scaler'] == 'mm':    
                min_=np.min(data[sensors])
                max_ =np.max(data[sensors])
                
                scaled_data=(((data[sensors]-min_)/(max_- min_))*(scalingparams['scaler_range'][1]-scalingparams['scaler_range'][0]))+scalingparams['scaler_range'][0]

                scalingparams['min_']=min_
                scalingparams['max_']=max_
            elif scalingparams['scaler'] == 'ss':
                mean_=np.mean(data[sensors])
                std_=np.std(data[sensors])

                scaled_data=(data[sensors]-mean_)/std_

                scalingparams['mean_']=mean_
                scalingparams['std_']=std_
                    
        else:    
            if scalingparams['scaler'] == 'mm':    
                min_=scalingparams['min_']
                max_=scalingparams['max_']
                scalingparams['scaler_range'] = scaler_range
                scaled_data=(((data[sensors]-min_)/(max_- min_))*(scalingparams['scaler_range'][1]-scalingparams['scaler_range'][0]))+scalingparams['scaler_range'][0]

            elif scalingparams['scaler'] == 'ss':
                mean_=scalingparams['mean_']
                std_=scalingparams['std_']
                
                scaled_data=(data[sensors]-mean_)/std_
                
        scaled_df = scaled_data
        
    scaled_df=scaled_df.dropna(axis=1)
    cols_wo_na = scaled_df.columns
    print("\n\t scaled_df after dropNA {} \n column names {}".format(scaled_df.shape, cols_wo_na))
    
    return scaled_df, cols_wo_na 

def sequence_length_generator(scaled_data, window_size, cols_wo_na):
    
    if "operating_condition" in scaled_data.columns:
        input_features= ["s2","s3","s4","s7","s8","s9", "s11","s12","s13","s14","s15", "s17","s20","s21"]#,"oc_0", "oc_1", "oc_2", "oc_3", "oc_4", "oc_5"]
    else:
        input_features= ["s2","s3","s4","s7","s8","s9", "s11","s12","s13","s14","s15", "s17","s20","s21"]
        
    groupby_scaled_data = scaled_data.groupby('engine_id',sort = False)
    
    sequences = []
    rul = []
    
    for engine_id, data in groupby_scaled_data:
        input_data = data[input_features]
        
        for p in range(window_size, input_data.shape[0]+1):  
            xtrain = input_data.iloc[p-window_size:p, 0:]
            xtrain = xtrain.to_numpy()
            sequences.append(xtrain)
            
            out_rul=data.iloc[p-1:p, -1]
            out_rul = np.array(out_rul)
            rul.append(out_rul)
    
    sequences = torch.FloatTensor(sequences)
    sequences = sequences.view(sequences.shape[0], 1, window_size, len(input_features)) #shape = [RUl , channel, window_size, feature]
    rul = torch.FloatTensor(rul)
    rul = rul.view(-1,1)
    
    return sequences, rul
     
def plot_sensors(sensor_data,scaled=True):
    
    sensors=['s1', 's2', 's3','s4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                       's15', 's16', 's17', 's18', 's19', 's20', 's21']

    sensor_plot = sensor_data.groupby('engine_id',sort = False)
    
    if scaled:
        title = 'Scaled Training Sensor Data Trajectory '
    else:
        title = 'Original Training Sensor Data Trajectory '
        
    for engine_id, data in sensor_plot:
            data = data[sensors]
            data.plot(subplots=True, figsize=(10, 10), layout = (7, 3), title = title + str(engine_id))
            plt.legend(loc='upper right')
            break

    return None

class train_dataset(Dataset):
    def __init__(self, read_path, sub_dataset, window_size, max_life, scaler, scaler_range, cluster=True):
        self.read_path = read_path
        self.sub_dataset = sub_dataset
        self.window_size = window_size
        self.max_life = max_life
        self.scaler = scaler
        self.scaler_range = scaler_range
        self.cluster = cluster
                
        self.train, self.train_rul = self.to_preprocess(self)
        
    def __getitem__(self, index):
        return self.train[index], self.train_rul[index]
     
    def __len__(self):
        return self.train.shape[0]
    
    @staticmethod 
    def to_preprocess(self):
        
        training_set = load_dataset(self.read_path, self.sub_dataset, self.max_life, self.cluster, train=True, test=False, finaltest=False)
        
        #Plotting sensor data
        plot_sensors(training_set, scaled=False)
        
        #Train Data Feature Scaling
        scaled_df, cols_wo_na = get_scaleing(training_set, self.scaler, self.scaler_range, train=True)
        training_set[cols_wo_na]=scaled_df
        print('\n\t Shape of scaled train data ' , training_set.shape)

        #Plotting scaled sensor data
        plot_sensors(training_set, scaled=True)
        #Train Data Time window
        train, train_rul = sequence_length_generator(training_set, self.window_size, cols_wo_na)
        print('\n\t The shape of Train sequences {} \n rul {}'.format(train.shape, train_rul.shape)) 
        
        return train, train_rul
    
class test_dataset(Dataset):
    def __init__(self, read_path, sub_dataset,  window_size, max_life, scaler, scaler_range, cluster=True):
        self.read_path =read_path
        self.sub_dataset = sub_dataset
        self.window_size = window_size
        self.max_life = max_life
        self.scaler = scaler
        self.scaler_range = scaler_range
        self.cluster=cluster
        
        self.test, self.test_rul = self.to_preprocess(self)

    def __getitem__(self, index):
            return self.test[index], self.test_rul[index]
         
    def __len__(self):
            return self.test.shape[0]
        
    @staticmethod 
    def to_preprocess(self):
        
        testing_set = load_dataset(self.read_path, self.sub_dataset, self.max_life, self.cluster, train=False, test=True, finaltest=False)
        
        #TestData Feature Scaling
        scaled_df, cols_wo_na = get_scaleing(testing_set, self.scaler, self.scaler_range, train=False)
        testing_set[cols_wo_na]=scaled_df
        print('\n\t Shape of scaled test data ' , testing_set.shape)

        #Test Data Time window
        test, test_rul = sequence_length_generator(testing_set, self.window_size, cols_wo_na)
        print('\n\t The shape of Test sequences {} \n rul {}'.format(test.shape, test_rul.shape)) 
        
        return test, test_rul

class finaltest_dataset(Dataset):
    def __init__(self, read_path, sub_dataset,  window_size, max_life, scaler, scaler_range, cluster=True):
        self.read_path = read_path
        self.sub_dataset = sub_dataset
        self.window_size = window_size
        self.max_life = max_life
        self.scaler = scaler
        self.scaler_range = scaler_range
        self.cluster=cluster
        self.final_test, self.final_test_rul  = self.to_preprocess(self)

    def __getitem__(self, index):
        return self.final_test[index]
     
    def __len__(self):
        return self.final_test.shape[0]
    
    @staticmethod 
    def to_preprocess(self):
        
        final_testing_set = load_dataset(self.read_path, self.sub_dataset, self.max_life, self.cluster, train=False, test=False, finaltest=True)

        #TestData Feature Scaling
        scaled_df, cols_wo_na = get_scaleing(final_testing_set, self.scaler, self.scaler_range, train=False)
        final_testing_set[cols_wo_na]=scaled_df
        print('\n\t Shape of scaled final_test data ' , final_testing_set.shape)

        #Test Data Time window
        final_test, final_test_rul = sequence_length_generator(final_testing_set, self.window_size, cols_wo_na)
        print('\n\t The shape of FianlTest sequences {} \n rul {}'.format(final_test.shape, final_test_rul.shape)) 
        
        return final_test,final_test_rul

################################################
################################################
################################################    
    
#CMPASS                   
def LoadCMAPSS(read_path, sub_dataset, window_size, max_life, scaler, scaler_range, shuffle, batch_size, cluster=True):
    
    if shuffle:
        
        tr_dataset = train_dataset(read_path, sub_dataset, window_size, max_life, scaler, scaler_range, cluster=True)
        print('\n\t Length of shuffeled train data: ' , len(tr_dataset))
        train_loader = DataLoader(tr_dataset, batch_size, shuffle=True)

        unshuffle_train_loader = DataLoader(tr_dataset, batch_size, shuffle=False)

    else:
        
        tr_dataset = train_dataset(read_path, sub_dataset,  window_size, max_life, scaler, scaler_range, cluster=True)
        print('\n\t Length of unshuffeled train data: ' , len(tr_dataset))
        train_loader = DataLoader(tr_dataset, batch_size, shuffle=False)
        
        unshuffle_train_loader = None

    te_dataset = test_dataset(read_path, sub_dataset,  window_size, max_life, scaler, scaler_range, cluster=True)
    print('\n\t Length of test data: ', len(te_dataset))
    test_loader = DataLoader(te_dataset, batch_size, shuffle=False)

    finalte_dataset = finaltest_dataset(read_path, sub_dataset,  window_size, max_life, scaler, scaler_range, cluster=True)
    print('\n\t Length of finaltest data: ' , len(finalte_dataset))
    finaltest_loader = DataLoader(finalte_dataset, batch_size, shuffle=False)

    return list(enumerate(train_loader)), list(enumerate(test_loader)), train_loader, test_loader, unshuffle_train_loader, finaltest_loader
