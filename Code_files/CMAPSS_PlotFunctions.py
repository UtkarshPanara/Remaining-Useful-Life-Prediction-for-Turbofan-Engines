# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:18:53 2020

@author: Utkarsh Panara
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt
import CMAPSS_Dataloader

def train_actual_predicted(read_path, sub_dataset, window_size, max_life, train_output, alpha_grid, alpha_low, alpha_high, _COLORS):
    
    train_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, train=True, test=False, finaltest=False)
    train_data_gp = train_data.groupby('engine_id', sort=False)
    train_data_reduce = []
    
    for engine_id, data in train_data_gp:
        data = data.iloc[window_size-1:,:]
        train_data_reduce.append(data)
        
    train_data = pd.concat(train_data_reduce, ignore_index=True)
    
    train_data['Predicted RUL'] = train_output
    
    train_rul_group = train_data.groupby('engine_id', sort = False)
    max_plots=[]
        
    for engine_id, data in train_rul_group:
        actual_rul = data[['cycle', 'RUL']]
        actual_rul = actual_rul.set_index('cycle')
        predicted_rul = data[['cycle','Predicted RUL']]
        predicted_rul = predicted_rul.set_index('cycle')
        
        fig=plt.figure(figsize=(7,7))
        ax=plt.subplot(1,1,1)
        plt.grid(b=True, which="both", axis="both", alpha=alpha_grid)
        plt.xlabel("Time (Cycle)", fontsize = 24)
        plt.ylabel("RUL", fontsize = 24)
        plt.xlim(data['cycle'].min(), data['cycle'].max())
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid",linewidth=4, label='Predicted RUL')
        plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
        ax.set_title('FD'+sub_dataset +" Train Engine Unit-" + str(engine_id), fontsize = 26)
        ax.legend(loc="upper right", fontsize = 16)
        max_plots.append(engine_id)
        
        if len(max_plots) >= 1:
            break

        fig.tight_layout()

    return None

def test_actual_predicted(read_path, sub_dataset, window_size, max_life, test_output, alpha_grid, alpha_low, alpha_high, _COLORS):
    
    test_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, train=False, test=True, finaltest=False)
    test_data_gp = test_data.groupby('engine_id', sort=False)
    test_data_reduce = []
    
    for engine_id, data in test_data_gp:
        data = data.iloc[window_size-1:,:]
        test_data_reduce.append(data)
        
    test_data = pd.concat(test_data_reduce, ignore_index=True)
    
    test_data['Predicted RUL'] = test_output
    
    test_rul_group = test_data.groupby('engine_id', sort = False)
    max_plots=[]
        
    for engine_id, data in test_rul_group:
        actual_rul = data[['cycle', 'RUL']]
        actual_rul = actual_rul.set_index('cycle')
        predicted_rul = data[['cycle','Predicted RUL']]
        predicted_rul = predicted_rul.set_index('cycle')
        
        fig=plt.figure(figsize=(7,7))
        ax=plt.subplot(1,1,1)
        plt.grid(b=True, which="both", axis="both", alpha=alpha_grid)
        plt.xlabel("Time (Cycle)", fontsize = 24)
        plt.ylabel("RUL", fontsize = 24)
        plt.xlim(data['cycle'].min(), data['cycle'].max())
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid",linewidth=4, label='Predicted RUL')
        plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
        ax.set_title('FD'+sub_dataset +" Test Engine Unit-" + str(engine_id), fontsize = 26)
        ax.legend(loc="upper right", fontsize = 16)
        max_plots.append(engine_id)
        
        if len(max_plots) >= 1:
            break

        fig.tight_layout()

    return None

def loss_plot(sub_dataset, train_loss_epoch, test_loss_epoch, alpha_grid, alpha_low, alpha_high, _COLORS):
   
   fig=plt.figure(figsize=(7,7))
   plt.grid(b=True, which="both", axis="both", alpha=alpha_grid)
   plt.xlabel('Epochs', fontsize=24)
   plt.ylabel('RMSE', fontsize=24)
   plt.xticks(fontsize=18)
   plt.yticks(fontsize=18)
   plt.plot(train_loss_epoch, color=_COLORS[0], alpha=alpha_low, linestyle="solid", label='Train Loss')
   plt.plot(test_loss_epoch, color=_COLORS[1], alpha=alpha_low, linestyle="solid", label='Test Loss')            
   fig.tight_layout()

   return None
  
def score_func(pred_rul, actual_rul):
    Si = []
    for i in range(0,pred_rul.shape[0]):
        di = pred_rul[i] - actual_rul[i]
        
        if di < 0:
            inter = torch.exp((-di)/13) - 1
        else:
            inter = torch.exp(di/10) - 1
        Si.append(inter)
    s = torch.sum(torch.stack(Si))
    return s

def fianltest_actual_vs_predicted(read_path, sub_dataset, window_size, max_life, finaltest_output, alpha_grid, 
                                  alpha_low, alpha_high, _COLORS):
    
    finaltest_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset,max_life, train=False, test=False, finaltest=True)
    finaltest_data_groupped = finaltest_data.groupby('engine_id', sort = False)
    finaltest_data_reduced = []

    for engine_id, data in finaltest_data_groupped:  
        data = data.iloc[window_size-1 : , :]
        finaltest_data_reduced.append(data)

    finaltest_data = pd.concat(finaltest_data_reduced, ignore_index=True)
    finaltest_data['Predicted RUL'] =  finaltest_output
    
    finaltest_rul_group = finaltest_data.groupby('engine_id', sort = False)
    
    if sub_dataset == "001":
        plot_ids=[24]
    if sub_dataset == "002":
        plot_ids=[5]
    if sub_dataset == "003":
        plot_ids=[3]
    if sub_dataset == "004":
        plot_ids=[32]

    for engine_id, data in finaltest_rul_group:
        if engine_id in plot_ids:
            actual_rul = data[['cycle', 'RUL']]
            actual_rul = actual_rul.set_index('cycle')
            predicted_rul = data[['cycle','Predicted RUL']]
            predicted_rul = predicted_rul.set_index('cycle')
            
            fig=plt.figure(figsize=(12,5))

            ax=plt.subplot(1,1,1)
            plt.grid(b=True, which="both", axis="both", alpha=alpha_grid)
            plt.xlabel("Time (Cycle)", fontsize=24)
            plt.ylabel("RUL",fontsize=24)
            plt.xlim(data['cycle'].min(), data['cycle'].max())
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4, label='Predicted RUL')
            plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
            ax.set_title('FD'+sub_dataset+" Final Test Engine Unit-" + str(engine_id), fontsize=26)
            ax.legend(loc="upper right", fontsize = 16)                  
            fig.tight_layout()
            
    rul_score = []
    for engine_id, data in finaltest_rul_group:
        last_rul = data["Predicted RUL"].iloc[-1]
        rul_score.append(last_rul)
    
    true_rul = pd.read_csv(read_path + "RUL_FD"+sub_dataset + ".csv")
    ture_rul = torch.FloatTensor(true_rul['RUL'].values)
    rul_socre =  torch.FloatTensor(rul_score)
    score = score_func(rul_socre, ture_rul)
    print("The Final Score is:", score.item())
     
    return score


