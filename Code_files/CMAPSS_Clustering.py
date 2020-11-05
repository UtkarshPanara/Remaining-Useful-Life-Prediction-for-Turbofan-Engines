# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:07:37 2020

@author: Utkarsh Panara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def cluster(read_path, sub_dataset):
    print("Initiating Clustering for FD{}".format(sub_dataset))
    #Read Files
    train_data = pd.read_csv(read_path+"train_FD"+sub_dataset+".csv")
    test_data = pd.read_csv(read_path+"test_FD"+sub_dataset+".csv")
    
    #Clustering
    input_kmeans = train_data[["setting1", "setting2", "setting3"]]
    
    estimator=KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=500, tol=0.0001, 
                 precompute_distances='auto', verbose=0, random_state=None, copy_x=True, 
                 n_jobs=None, algorithm='auto')
    
    estimator.fit(input_kmeans)
    train_data_labels = estimator.labels_
    
    test_data_labels = estimator.predict(test_data[["setting1", "setting2", "setting3"]])

    #Visualizing Clusters
    fig = plt.figure(dpi=100)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(test_data['setting1'],test_data['setting2'],test_data['setting3'],
               c=test_data_labels.astype(np.float),cmap='Greens', edgecolor='k', s=150, marker='^')
    
    ax.set_xlabel('Operating Setting 1')
    ax.set_ylabel('Operating Setting 2')
    ax.set_zlabel('Operating Setting 3')
    ax.set_title('FD'+sub_dataset+' Test Data Operating Modes', fontsize=20)
    ax.dist = 14
    plt.show()
    
    #Adding labels column to the main dataframe
    train_data["operating_condition"] = train_data_labels
    test_data["operating_condition"] = test_data_labels
    
    print("Complete Clustering for FD{} and added a column named operating_condition".format(sub_dataset))

    return train_data, test_data

def oc_history_cols(read_path, sub_dataset, train_data, test_data, save=False):
    
    if "operating_condition" not in train_data.columns or "operating_condition" not in test_data.columns:
        print("Column operating_condition is not found in the data frame")

    else:
        print("Adding History Columns in the Data Frame")
        train_data[["oc_0","oc_1","oc_2","oc_3","oc_4","oc_5"]]= pd.DataFrame([[0,0,0,0,0,0]], index=train_data.index)
        test_data[["oc_0","oc_1","oc_2","oc_3","oc_4","oc_5"]]= pd.DataFrame([[0,0,0,0,0,0]], index=test_data.index)
        
        for file in ["train", "test"]:
            if file == "train":
                groupby_traj = train_data.groupby('engine_id', sort=False)
            else:
                groupby_traj = test_data.groupby('engine_id', sort=False)
                
            additional_oc=[]
            for engine_id, data in groupby_traj:
                data=data.reset_index()
                for i in range(data.shape[0]):
                    check_oc=data.iloc[i]["operating_condition"]
                    if  i != data.shape[0]-1:
                        data.at[i+1:, "oc_"+str(int(check_oc))]=data.iloc[i+1]["oc_"+str(int(check_oc))]+1
                additional_oc.append(data)
            
            oc_cols=pd.concat(additional_oc,  sort=False, ignore_index=False)
            oc_cols=oc_cols.set_index('index', drop=True)
            
            if save:
                print("Saving the {} data with operating condition history columns".format(file))
                oc_cols.to_csv(read_path+file+"_FD"+sub_dataset+"_cluster.csv", index=False)
    return None







