# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:26:57 2020

@author: Utkarsh Panara
"""
import torch

class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.zeropad = torch.nn.ZeroPad2d((0,0,0,9))
        self.conv1 = torch.nn.Conv2d(1, 10, (10,1), 1,0,1)
        self.conv2 = torch.nn.Conv2d(10,10,(10,1),1,0,1)
        self.conv3 = torch.nn.Conv2d(10,10,(10,1),1,0,1)
        self.conv4 = torch.nn.Conv2d(10,10,(10,1),1,0,1)
        self.conv5 = torch.nn.Conv2d(10,1,(3,1),1,(1,0),1)
        self.fc1 = torch.nn.Linear(420,100)
        self.fc2 = torch.nn.Linear(100,1)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.activfunc = torch.nn.Tanh()
        
    def forward(self,input_):
        
        out = self.zeropad(input_)
        out = self.conv1(out)
        out = self.activfunc(out)

        out = self.zeropad(out)
        out = self.conv2(out)
        out = self.activfunc(out)

        out = self.zeropad(out)
        out = self.conv3(out)
        out = self.activfunc(out)

        out = self.zeropad(out)
        out = self.conv4(out)
        out = self.activfunc(out)
        
        out = self.conv5(out)
        out = self.activfunc(out)

        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        out = self.activfunc(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    
def weights_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out')
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)
    
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out')
        if layer.bias is not None:
            layer.bias.data.fill_(0.001)
            
    return None

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
            
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred,actual))
            
def Network(algorithm = "CNN"):

    if algorithm == "CNN":
        model = CNN()
        model.apply(weights_init)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-04, weight_decay=0)
    loss_func = RMSELoss()
    
    return model, optimizer, loss_func
    
    
    
    
    
    