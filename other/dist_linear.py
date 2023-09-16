import torch
import torchvision
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F


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

class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
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
target_model = SplitNN().to(device=device)

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
optimiser=torch.optim.SGD(target_model.parameters(),lr=0.001,momentum=0.9)
cost = torch.nn.CrossEntropyLoss()


def attack_test(train_loader, target_model, attack_model):
    model = SplitNN()
    psnr_lst, ssim_lst, fid_lst=[], [], []
    correct=0
    attack_correct=0
    total=0
    for batch, (data, targets) in enumerate(tqdm(train_loader)):
        #data = data.view(data.size(0), -1)
        data, targets = data.to(device=device), targets.to(device=device)
        #org_data=data
        #data= data.view(-1,32*32*3)
        target_outputs = target_model.first_part(data)
        #recreated_data = attack_model(target_outputs)
        print(data.shape)
        print(target_outputs.shape)
        #gen_data= recreated_data.view(-1,32*32*3)
        
        target_outputs = target_outputs[0] / 2 + 0.5
        DataI=target_outputs[:,:, 0:31]
        img= torch.permute(DataI, (2, 1,0))
        #img=img.to(torch.float32)
        #print(img.shape)
        plt.imshow((img.cpu().detach().numpy()))
        plt.xticks([])
        plt.yticks([])

        plt.draw()
        plt.savefig(f'./distribution/img/linear/img{batch}.jpg', dpi=100, bbox_inches='tight')

        
        
    
    return img

def plot_dist():
    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)

    for image in os.listdir('./distribution/img/linear/'):
        img = Image.open('./distribution/img/linear/'+image)
        x = np.array(img)
        x = x.transpose(2, 0, 1)
        hist_r = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        hist_g = np.histogram(x[1], bins=nb_bins, range=[0, 255])
        hist_b = np.histogram(x[2], bins=nb_bins, range=[0, 255])
        count_r += hist_r[0]
        count_g += hist_g[0]
        count_b += hist_b[0]

    bins = hist_r[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count_r, color='r', alpha=0.7)
    plt.bar(bins[:-1], count_g, color='g', alpha=0.7)
    plt.bar(bins[:-1], count_b, color='b', alpha=0.7)
    plt.savefig(f'./distribution/linear_dist.jpg', dpi=100, bbox_inches='tight')
    
    return fig
#target_epochs=50
loss_train_tr, loss_test_tr=[],[]
#attack_epochs=100

loss_train, loss_test=[],[]

print("**********Test Starting************")
#img=attack_test(train_loader, target_model, attack_model)
fig=plot_dist()

print('Done!')