# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:13:22 2020

@author: Utkarsh Panara
"""

def Train(train_loader, test_loader, unshuffle_train_loader, finaltest_loader, model, optimizer, loss_func, num_epochs=100):
    
    train_loss_epoch = []
    test_loss_epoch = []
    train_output = []
    test_output = []
    finaltest_output = []
    print("\tEpoch | \tTrain Loss | \tTest Loss")
    for epoch in range(num_epochs):
        
        running_loss_tr = 0
        running_loss_te = 0
        batch_counter_tr = 0
        batch_counter_te = 0
        model.train()

        for i, (data_tr, label_tr) in enumerate(train_loader):
            batch_counter_tr += 1
            optimizer.zero_grad()
            output_tr = model(data_tr.float())
            loss_tr = loss_func(output_tr, label_tr)
            loss_tr.backward()
            optimizer.step()
            running_loss_tr += loss_tr.item()
            
            if epoch == num_epochs-1:
                train_output += output_tr.flatten().tolist()
                
        epoch_loss_tr = running_loss_tr / batch_counter_tr
        train_loss_epoch.append(epoch_loss_tr)
        
        if epoch == num_epochs-1:
            if unshuffle_train_loader is not None:
                train_output = []
                for i, (data_uns_tr, lable_un_tr) in enumerate(unshuffle_train_loader):
                    output_uns_tr = model(data_uns_tr.float())
                    train_output += output_uns_tr.flatten().tolist()
            else:
                pass
        else:
            pass

        model.eval()
        for i, (data_te, label_te) in enumerate(test_loader):
            batch_counter_te +=1
            output_te = model(data_te.float())
            loss_te = loss_func(output_te, label_te)
            running_loss_te += loss_te.item()
            
            if epoch == num_epochs-1:
                test_output += output_te.flatten().tolist()
            else:
                pass
            
        epoch_loss_te = running_loss_te / batch_counter_te
        test_loss_epoch.append(epoch_loss_te)
        print("\n\t{} \t{} \t{}".format(epoch+1, epoch_loss_tr, epoch_loss_te))
        
    model.eval()
    for i, (data_fte) in enumerate(finaltest_loader):
        output_fte= model(data_fte.float())
        finaltest_output += output_fte.flatten().tolist()
    
    return train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output




