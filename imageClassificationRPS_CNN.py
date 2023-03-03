import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage, RandomRotation
from torch.utils.data import DataLoader, Dataset,random_split
from torchsummary import summary
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve


device = "cuda" if torch.cuda.is_available() else "cpu"

def testModel(model,testImages,normTransform,label):    
    out = model(normTransform(testImages).to(device))
    out=F.softmax(out)
    preidctions=np.argmax(out.detach().cpu().numpy(),1)
    predictedClasses=np.array(trainSet.classes)[preidctions]
    GTClasses=np.array(trainSet.classes)[label]
    ################################################## display a batch
    plt.figure()
    for i in range(6):
        pilImge=ToPILImage()(testImages[i])
        plt.subplot(2,3,i+1)
        plt.imshow(pilImge)
        plt.title(predictedClasses[i]+'_'+GTClasses[i])
    plt.show(block=True)



#transforms
normTransform=Normalize(mean=torch.Tensor([0.8502, 0.8215, 0.8116]),std=torch.Tensor([0.2089, 0.2512, 0.2659]))
transform=Compose([Resize(28),RandomRotation(90), ToTensor(), normTransform])
#Datasets & Loaders
trainSet=ImageFolder(root='dataset/rps/',transform=transform)
train_loader=DataLoader(trainSet, batch_size=128, shuffle=True)
valSet=ImageFolder(root='dataset/rps-test-set/',transform=transform)
val_loader=DataLoader(valSet, batch_size=128)


transform2=Compose([Resize(28), ToTensor()])
testSet=ImageFolder(root='dataset/rps-test-set/',transform=transform2)
test_loader=DataLoader(testSet, batch_size=6,shuffle=True)
################################################## display a batch
testImages,label=next(iter(test_loader))
GTClasses=np.array(trainSet.classes)[label]
for i in range(6):
    pilImge=ToPILImage()(testImages[i])
    plt.subplot(2,3,i+1)
    plt.imshow(pilImge)
    plt.title(GTClasses[i])
    
plt.show(block=True)
#####################################################


class RPSModel(nn.Module): #defining your custom model class
    def __init__(self, n_filters):
        super().__init__()

        self.n_filters = n_filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_filters,kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)        
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=self.n_filters, kernel_size=3)        
        
        self.flatten = nn.Flatten()    
        self.linear1 = nn.Linear(in_features=self.n_filters*5*5, out_features=50)
        self.linear2 = nn.Linear(50, 3)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        #Featurizer
        x = self.conv1(X)
        x = self.relu(x)
        x= self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #Classifier
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)        
        x = self.dropout(x)
        x = self.linear2(x)
        return x

model = RPSModel(16).to(device) #our custom model
num_params = sum(param.numel() for param in model.parameters())
num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(model)
summary(model,(3,28,28))

#testing model befre training on some test images
testModel(model,testImages,normTransform,label)




#optimizer = optim.SGD(model.parameters(), lr=lr)
lr=3e-4
optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-08)
loss_fn = nn.CrossEntropyLoss()

#tensorboard 
tboardWriter=SummaryWriter('runs/RPSClassification-CNN')


#batch wise training loop
epochs = 20
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
        logits = model(X_train)               
        
        loss = loss_fn(logits, Y_train)       
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #store metrics for all batches of current epoch 
        y_hat=F.softmax(logits,dim=-1)
        y_hat=y_hat.detach().cpu().numpy()
        y_hat=np.argmax(y_hat,axis=1)
        y_hat=y_hat.reshape(-1,1)

        Y_train=Y_train.detach().cpu().numpy()
        Y_train=Y_train.reshape(-1,1)
        all_Y_train_epoch=np.vstack((all_Y_train_epoch,Y_train))
        all_Yhat_train_epoch=np.vstack((all_Yhat_train_epoch,y_hat))   
        all_train_losses_epoch=np.append(all_train_losses_epoch,loss.item())     
        
    
    
    #computing metrics for current epoch
    train_losses.append(all_train_losses_epoch.mean()) #mean loss for all batches    
    acTrain=accuracy_score(all_Y_train_epoch, all_Yhat_train_epoch)
    cmTrain=confusion_matrix(all_Y_train_epoch, all_Yhat_train_epoch)
    print(cmTrain)

    #validation loop also bacth wise
    all_Y_val_epoch=np.array([]).reshape(0,1)
    all_Yhat_val_epoch=np.array([]).reshape(0,1)
    all_val_losses_epoch=np.array([])
    for X_val, Y_val in val_loader:  #batch wise validation set predictions only
        model.eval()
        
        X_val = X_val.to(device)
        Y_val = Y_val.to(device)
        
        with torch.no_grad():            
            logits = model(X_val)           
            loss = loss_fn(logits, Y_val)
        
        #store metrics for all batches of current epoch 
        y_hat_val=F.softmax(logits,dim=-1)
        y_hat_val=y_hat_val.detach().cpu().numpy()
        y_hat_val=np.argmax(y_hat_val,axis=1)
        y_hat_val=y_hat_val.reshape(-1,1)
        Y_val=Y_val.detach().cpu().numpy()
        Y_val=Y_val.reshape(-1,1)
        all_Y_val_epoch=np.vstack((all_Y_val_epoch,Y_val))
        all_Yhat_val_epoch=np.vstack((all_Yhat_val_epoch,y_hat_val))   
        all_val_losses_epoch=np.append(all_val_losses_epoch,loss.item())     
            

    #computing metrics for current epoch
    val_losses.append(all_val_losses_epoch.mean()) #mean loss for all batches    
    acVal=accuracy_score(all_Y_val_epoch, all_Yhat_val_epoch)
    cmVal=confusion_matrix(all_Y_val_epoch, all_Yhat_val_epoch)
    
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





#testing model after training on some test images
testModel(model,testImages,normTransform,label)



