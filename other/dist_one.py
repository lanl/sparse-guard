import torch
import torchvision
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from lcapt.lca import LCAConv2D


n_epochs = 3
batch_size_train = 64
batch_size_attack=1
batch_size_test = 1



train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('/vast/home/sdibbo/def_ddlc/data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('/vast/home/sdibbo/def_ddlc/data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_test, shuffle=True)
'''
attack_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('/vast/home/sdibbo/def_ddlc/data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
  batch_size=batch_size_attack, shuffle=True)
'''  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

class SplitNN_linear(nn.Module):
  def __init__(self):
    super(SplitNN_linear, self).__init__()
    self.first_part = nn.Sequential(
                           nn.Linear(32, 500),
                           nn.ReLU(),
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(48000, 500),
                           nn.ReLU(),
                           nn.Linear(500, 10),
                           nn.Softmax(dim=-1),
                         )
class SplitNN_cnn(nn.Module):
  def __init__(self):
    super(SplitNN_cnn, self).__init__()
    self.first_part = nn.Sequential(
       nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(), 
                           nn.Linear(32, 500),
                           nn.ReLU(),
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(48000, 500),
                           nn.ReLU(),
                           nn.Linear(500, 10),
                           nn.Softmax(dim=-1),
                         )
class SplitNN_lca(nn.Module):
  def __init__(self):
    super(SplitNN_lca, self).__init__()
    self.first_part = nn.Sequential(
       LCAConv2D(out_neurons=16,
                in_neurons=3,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.4, lca_iters=500, pad="same", dtype=torch.float16,                
            ),  
            nn.BatchNorm2d(16),                           
            LCAConv2D(out_neurons=32,
                in_neurons=16,                        
                kernel_size=5,              
                stride=1,                   
                 lambda_=0.4, lca_iters=500, pad="same", dtype=torch.float16),  
                 nn.BatchNorm2d(32),  
                          nn.Linear(32, 500),
                           nn.ReLU(),
 
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(48000, 500),
                           nn.ReLU(),
                           nn.Linear(500, 10),
                           nn.Softmax(dim=-1),
                         )  
  def forward(self, x):
    #x=x.view(-1,32*32*3)
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = x.view(-1, 48000)
    #print(x.shape)
    x=self.second_part(x)
    return x

'''
class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.first_part = nn.Sequential(
                           nn.Linear(3*32*32,512),
                           nn.ReLU(),
                           #nn.Dropout(0.25),
                           #nn.Linear(2048,512),
                           #nn.ReLU(),
                           #nn.Dropout(0.25),
                           #nn.Linear(1024,512),
                           #nn.ReLU(),
                           #nn.Dropout(0.25),
                         )
    self.second_part = nn.Sequential(
                           nn.Linear(512, 256),
                           nn.ReLU(),
                           #nn.Dropout(0.25),
                           nn.Linear(256, 10),
                           #nn.Softmax(dim=-1),
                         )

  def forward(self, x):
    x=x.view(-1,32*32*3)
    x=self.first_part(x)
    #print(x.shape)
    #x = torch.flatten(x, 1) # flatten all dimensions except batch
    #x = x.view(-1, 48000)
    #print(x.shape)
    x=self.second_part(x)
    return x
'''    
target_model1 = SplitNN_linear().to(device=device)
target_model2 = SplitNN_cnn().to(device=device)
target_model3 = SplitNN_lca().to(device=device, dtype=torch.float32)

class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                      nn.Linear(500, 800),
                      nn.ReLU(),
                      nn.Linear(800, 32),
                    )
 
  def forward(self, x):
    return self.layers(x)
'''
class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
                      nn.Linear(512, 800),
                      nn.ReLU(),
                      nn.Linear(800, 32*32*3),
                    )
 
  def forward(self, x):
    return self.layers(x)  
'''    
attack_model = Attacker().to(device=device)
#optimiser = optim.Adam(target_model.parameters(), lr=1e-4)
optimiser=torch.optim.SGD(target_model1.parameters(),lr=0.001,momentum=0.9)
cost = torch.nn.CrossEntropyLoss()


def attack_test(train_loader, target_model1, target_model2, target_model3):
    #model = SplitNN()
    psnr_lst, ssim_lst, fid_lst=[], [], []
    correct=0
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        data, targets = data.to(device=device), targets.to(device=device)
        #org_data=data
        #data= data.view(-1,32*32*3)
        if (batch==0):
            target_outputs1 = target_model1.first_part(data)
            target_outputs2 = target_model2.first_part(data)
            data, targets = data.to(device=device, dtype=torch.float32), targets.to(device=device)
            target_outputs3 = target_model3.first_part(data)
            #recreated_data = attack_model(target_outputs)
            DataI = data[0] / 2 + 0.5
            img= torch.permute(DataI, (1,2, 0))
            #img=img.to(torch.float32)
            #print(img.shape)
            plt.imshow((img.cpu().detach().numpy()))
            plt.xticks([])
            plt.yticks([])

            plt.draw()
            plt.savefig(f'./distribution/one/org_img{batch}.jpg', dpi=100, bbox_inches='tight')



            #gen_data= recreated_data.view(-1,32*32*3)
            
            target_outputs1 = target_outputs1[0] / 2 + 0.5
            DataI1=target_outputs1[:,:, 0:31]
            img1= torch.permute(DataI1, (2, 1,0))
            #img=img.to(torch.float32)
            #print(img.shape)
            plt.imshow((img1.cpu().detach().numpy()))
            plt.xticks([])
            plt.yticks([])

            plt.draw()
            plt.savefig(f'./distribution/one/lin_img{batch}.jpg', dpi=100, bbox_inches='tight')

            target_outputs2 = target_outputs2[0] / 2 + 0.5
            DataI2=target_outputs2[:,:, 0:3]
            #img= torch.permute(DataI, (2, 1,0))
            #img=img.to(torch.float32)
            #print(img.shape)
            plt.imshow((DataI2.cpu().detach().numpy()))
            plt.xticks([])
            plt.yticks([])

            plt.draw()
            plt.savefig(f'./distribution/one/cnn_img{batch}.jpg', dpi=100, bbox_inches='tight')
            

            target_outputs3 = target_outputs3[0] / 2 + 0.5
            DataI3=target_outputs3[:,:, 0:3]
            #img= torch.permute(DataI, (2, 1,0))
            #img=img.to(torch.float32)
            #print(img.shape)
            plt.imshow((DataI3.cpu().detach().numpy()))
            plt.xticks([])
            plt.yticks([])

            plt.draw()
            plt.savefig(f'./distribution/one/lca_img{batch}.jpg', dpi=100, bbox_inches='tight')
            
            
    
    return img, img1, DataI2, DataI3

def plot_dist():
    image1 = io.imread('./distribution/one/lin_img0.jpg')
    image2 = io.imread('./distribution/one/cnn_img0.jpg')
    image3 = io.imread('./distribution/one/lca_img0.jpg')

    plt.figure(figsize=(10,6))
    _ = plt.hist(image1.ravel(), bins = 256, color = 'orange', )
    plt.hist(image1[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    plt.hist(image1[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    plt.hist(image1[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()
    plt.savefig(f'./distribution/one/linear_dist.jpg', dpi=100, bbox_inches='tight')
    
    _ = plt.hist(image2[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    _ = plt.hist(image2[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    _ = plt.hist(image2[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.draw()
    plt.savefig(f'./distribution/one/cnn_dist.jpg', dpi=100, bbox_inches='tight')
    
    _ = plt.hist(image3[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    _ = plt.hist(image3[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    _ = plt.hist(image3[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.draw()
    plt.savefig(f'./distribution/one/lca_dist.jpg', dpi=100, bbox_inches='tight')
    
    return image1, image2, image3
#target_epochs=50
loss_train_tr, loss_test_tr=[],[]
#attack_epochs=100

loss_train, loss_test=[],[]

print("**********Test Starting************")
img1, img2, img3, img4=attack_test(train_loader, target_model1, target_model2, target_model3)
fig1, fig2, fig3=plot_dist()

print('Done!')