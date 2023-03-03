#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset,Dataset
from torchsummary import summary
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve

from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(13)
device='cuda' if torch.cuda.is_available() else 'cpu'

class SimpleClassificationNet(torch.nn.Module):
            
    def __init__(self):
        super().__init__()        
        self.linearLayer1 = nn.Linear(64,32) #hidden layer 
        self.s1=nn.Sigmoid()
        self.linearLayer2 = nn.Linear(32,16)
        self.s2=nn.Sigmoid()
        self.linearLayer3 = nn.Linear(16,1)
        self.s3=nn.Sigmoid()


    def forward(self,x):
        u=self.linearLayer1(x)
        v=self.s1(u)
        w=self.linearLayer2(v)
        x=self.s2(w)
        z=self.linearLayer3(x)
        yhat=self.s3(z)
        return yhat


class digits_datasets(Dataset):
    def __init__(self,x_tensor, y_tensor):
        super().__init__()
        self.X=x_tensor
        self.Y=y_tensor
        

    def __getitem__(self, index):
        return (self.X[index],self.Y[index])
    
    def __len__(self):
        return len(self.X)




def figure1(X_train, y_train, X_val, y_val, cm_bright=None):
    if cm_bright is None:
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)#, edgecolors='k')
    ax[0].set_xlabel(r'$X_1$')
    ax[0].set_ylabel(r'$X_2$')
    ax[0].set_xlim([-2.3, 2.3])
    ax[0].set_ylim([-2.3, 2.3])
    ax[0].set_title('Generated Data - Train')

    ax[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright)#, edgecolors='k')
    ax[1].set_xlabel(r'$X_1$')
    ax[1].set_ylabel(r'$X_2$')
    ax[1].set_xlim([-2.3, 2.3])
    ax[1].set_ylim([-2.3, 2.3])
    ax[1].set_title('Generated Data - Validation')
    fig.tight_layout()
    
    return fig


#Getting a toy dataset from scikit learn library
digits = load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y=digits.target
y=np.array([1 if i>0 else 0 for i in y])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=13)
#preprocessing normalizing the features (mean=0, var=1)
sc = StandardScaler()
sc.fit(X_train)  #note only from training data

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

#plotting scaled data
fig = figure1(X_train, y_train, X_val, y_val)
plt.show()
print(digits.data.shape)

plt.gray()
plt.matshow(digits.images[0])

plt.show()
#Preparing PyTorch DataSets and DataLoaders

# Builds tensors from numpy arrays
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()
x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()
# Builds dataset containing ALL data points
train_dataset = digits_datasets(x_train_tensor, y_train_tensor)
val_dataset = digits_datasets(x_val_tensor, y_val_tensor)
# Builds a loader of each set
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)
test_batch=next(iter(train_loader))
total_batches_one_epoch = len(iter(train_loader))



#model, optimizer and loss
model = SimpleClassificationNet().to(device)
stateDict=model.state_dict()
# print(stateDict)
# print(model)
# summary(model,(1,2))

lr = 0.001
optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.CrossEntropyLoss ()

#tensorboard 
tboardWriter=SummaryWriter('runs/simpleClassification')


#batch wise training loop
epochs = 1000
train_losses = []
val_losses = []
best_accuracy=0
for epoch in range(epochs):  #epochs loop

    all_Y_train_epoch=np.array([]).reshape(0,1)
    all_Yhat_train_epoch=np.array([]).reshape(0,1)
    all_train_losses_epoch=np.array([])

    for X_train, Y_train in train_loader:        #batch wise  training on train set
        model.train()
        X_train = X_train.to(device)
        Y_train = Y_train.to(device) 
#         print(X_train.shape)
        y_hat = model(X_train)  
#         print(y_hat.shape)
        
        loss = loss_fn(y_hat, Y_train)       
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #store metrics for all batches of current epoch 
        all_Y_train_epoch=np.vstack((all_Y_train_epoch,Y_train.detach().cpu().numpy()))
        all_Yhat_train_epoch=np.vstack((all_Yhat_train_epoch,y_hat.detach().cpu().numpy()))   
        all_train_losses_epoch=np.append(all_train_losses_epoch,loss.item())     
        
    
    
    #computing metrics for current epoch
    train_losses.append(all_train_losses_epoch.mean()) #mean loss for all batches
    preidctions=(all_Yhat_train_epoch>=0.5) #from probabilities to predictions
    acTrain=accuracy_score(all_Y_train_epoch, preidctions)


    #validation loop also bacth wise
    all_Y_val_epoch=np.array([]).reshape(0,1)
    all_Yhat_val_epoch=np.array([]).reshape(0,1)
    all_val_losses_epoch=np.array([])
    for X_val, Y_val in val_loader:  #batch wise validation set predictions only
        model.eval()
        
        X_val = X_val.to(device)
        Y_val = Y_val.to(device)
        
        with torch.no_grad():            
            y_hat_val = model(X_val)           
            loss = loss_fn(y_hat_val, Y_val)
        
        #store metrics for all batches of current epoch 
        all_Y_val_epoch=np.vstack((all_Y_val_epoch,Y_val.detach().cpu().numpy()))
        all_Yhat_val_epoch=np.vstack((all_Yhat_val_epoch,y_hat_val.detach().cpu().numpy()))   
        all_val_losses_epoch=np.append(all_val_losses_epoch,loss.item())     
            

    #computing metrics for current epoch
    val_losses.append(all_val_losses_epoch.mean()) #mean loss for all batches
    preidctions=(all_Yhat_val_epoch>=0.5) #from probabilities to predictions
    acVal=accuracy_score(all_Y_val_epoch, preidctions)

    print(f"epoch= {epoch}, accuracyTrain= {acTrain}, accuracyVal= {acVal}, train_loss= {train_losses[epoch]}, validation_loss= {val_losses[epoch]}")

    #checkpointing training
    if(acVal>best_accuracy):
        checkpoint = {'epoch': epoch,'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),'loss': train_losses,
                      'val_loss': val_losses}
        torch.save(checkpoint,'best.pth')




    tboardWriter.add_scalar("Loss/train", train_losses[epoch], epoch)
    tboardWriter.add_scalar("Loss/val", val_losses[epoch], epoch)
    tboardWriter.add_scalar("accuracy/train", acTrain, epoch)
    tboardWriter.add_scalar("accuracy/val", acVal, epoch)


#loading best model
checkpoint = torch.load('best.pth')
# Restore state for model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
total_epochs = checkpoint['epoch']
losses = checkpoint['loss']
val_losses = checkpoint['val_loss']


# In[ ]:




