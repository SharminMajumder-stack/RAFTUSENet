# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:45:08 2024

@author: sharminjouty48
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:11:42 2023

@author: sharminjouty48
"""
import sys
#sys.path.append('core_functions')
sys.path.append('Networks')
sys.path.append('Functions')
#sys.path.append('Networks_CNN')
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft_Large_USE_edited3 import RAFT
#from raft_Large_USE_edited import RAFT
#from raft_Large_USE import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import elastography
#import elastography_RAFT_USENetPaper
#from Configure import train_dataset_params, val_dataset_params,model_params,visualization_params, freq_params
from scipy.io import savemat
import scipy.io as sio

import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import matplotlib.pyplot as plt # plotting library
#import pandas as pd 
import random 
from torch.utils.data import DataLoader,random_split
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.set_device(0)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from scipy import signal
from scipy.ndimage import gaussian_filter
import torch.optim
import torch.utils.data
from torch.autograd import Variable
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

        
# exclude extremly large displacements
MAX_FLOW = 1000
SUM_FREQ = 100
VAL_FREQ = 5000
from scipy.io import savemat 
#%% Importing dataset
dataset_dir=r"D:\OneDrive - Texas A&M University\TAMU\Research\ML_DL_AI\Elastography\Data\Sharmin\MyData1"

parser = argparse.ArgumentParser(description='Ultrasound elastography using different networks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default=dataset_dir,
                    help='path to dataset')
parser.add_argument('--dataset', metavar='elastography', default='elastography_data',
                    choices='dataset_names',
                    help='dataset type : ' +
                    ' | '.join('dataset_names'))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.1, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument(
    "--seed_split",
    type=int,
    default=None,
    help="Seed the train-val split to enforce reproducibility (consistent restart too)",
)


parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--epoch-size', default=5, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')

global args, best_loss
best_loss=1000.0
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

up_size = [2048,512] ## simulation data size
#up_size = [1792,416] ## phantom data size

#up_size = (([up_size[0]//2,up_size[1]//2]))
   
### every time run this part of code to make  reproducibility
#torch.manual_seed(0)
def seed_all(seed):
    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_all(0)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
g = torch.Generator()
g.manual_seed(0)



print("=> fetching img pairs in '{}'".format(args.data))

train_set, test_set = elastography.elastography_TrainTestData(
        args.data,
        transform=None,
        target_transform=None,
        co_transform=None,
        split=args.split_file if args.split_file else args.split_value
    )


print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                            len(train_set),
                                                                            len(test_set)))

train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)


#from torchvision import transforms

Resize_transform = T.Compose([
       # T.Resize((984,236))
        T.Resize((up_size))
    ])  

Resize_init_flo = T.Compose([
       # T.Resize((984,236))
        T.Resize(([up_size[0]//8,up_size[1]//8]))
    ]) 



target_transform = T.Compose([
        T.Resize((up_size)),
    ])


def image_normalize(Im1,Im2):
    
   min_val = Im1.min(-1)[0].min(-1)[0]
   max_val = Im1.max(-1)[0].max(-1)[0]
   img1_batch = (Im1-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
   
   min_val = Im2.min(-1)[0].min(-1)[0]
   max_val = Im2.max(-1)[0].max(-1)[0]
   img2_batch = (Im2-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
   
   return img1_batch, img2_batch



def target_normalize(Im1):
    
   min_val = Im1.min(-1)[0].min(-1)[0]
   max_val = Im1.max(-1)[0].max(-1)[0]
   img1_batch = (Im1-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
   #img1_batch = (Im1-min_val[:,:,None,None])
     
   return img1_batch

# exclude extremly large displacements
MAX_FLOW = 1000
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

import losses
import ImageProcess
def sequence_NCC(flow_preds, Im1, Im2,valid=0.9, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    warped_img = ImageProcess.warp_image(Im2[:,1:2,:,:], flow_preds)
        
    flow_similarity = (losses.LNCC(warped_img, Im1[:,1:2,:,:]))
        
        #losses.NCC(warped_img, Im1)
        #flow_similarity += i_weight * (valid[:, None] * i_similarity).mean()
       #flow_similarity += i_weight * ( i_similarity).mean()
        
    return 1.0-(flow_similarity)


def sequence_Dataloss(flow_preds, Im1, Im2,valid=0.9, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    warped_img = ImageProcess.warp_image(Im2, flow_preds)
    i_loss = (warped_img - Im1).abs()
        #epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    flow_loss =  (i_loss).mean()


    return flow_loss

def sequence_gradNorm(flow_preds):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    
    for i in range(n_predictions):
        flow_loss=losses.GradNorm(flow_preds[i]).mean()

    return flow_loss/n_predictions



def strainCompatibilityLoss(Ax_disp,flow_preds):
    flow_loss = 0.0
    if isinstance(flow_preds, list):
       n_predictions = len(flow_preds)
       seq_preds=flow_preds
    else:
        n_predictions=1

    for i in range(n_predictions):
        if isinstance(flow_preds, list):
           Lat_disp=flow_preds
        else : 
        
            Lat_disp=flow_preds[i][None,:]
          
        i_weight = 0.95**(n_predictions - i - 1)
        strainAx = torch.abs(Ax_disp[:, :, :-2, :-2]-Ax_disp[:, :, 2:, :-2]+Ax_disp[:, :, :-2, 2:]-Ax_disp[:, :, 2:, 2:])
        strainLat = torch.abs(Lat_disp[:, :, :-2, :-2]-Lat_disp[:, :, 2:, 2:]+Lat_disp[:, :, 2:, :-2]-Lat_disp[:, :, 2:, 2:])
        strainAxShear = torch.abs(Ax_disp[:, :, :-2, :-2]-Ax_disp[:, :, 2:, 2:]+Ax_disp[:, :, 2:, :-2]-Ax_disp[:, :, 2:, 2:])
        strainLatShear = torch.abs(Lat_disp[:, :, :-2, :-2]-Lat_disp[:, :, 2:, :-2]+Lat_disp[:, :, :-2, 2:]-Lat_disp[:, :, 2:, 2:])

        ShearStrain=(strainAxShear+strainLatShear)/2
        dLatdY = torch.abs(strainLat[:, :, :-2, :-2]-strainLat[:, :, 2:, :-2]+strainLat[:, :, :-2, 2:]-strainLat[:, :, 2:, 2:])
        dLatdYY = torch.abs(dLatdY[:, :, :-2, :-2]-dLatdY[:, :, 2:, :-2]+dLatdY[:, :, :-2, 2:]-dLatdY[:, :, 2:, 2:])
        dAxdX = torch.abs(strainAx[:, :, :-2, :-2]-strainAx[:, :, 2:, 2:]+strainAx[:, :, 2:, :-2]-strainAx[:, :, 2:, 2:])
        dAxdXX = torch.abs(dAxdX[:, :, :-2, :-2]-dAxdX[:, :, 2:, 2:]+dAxdX[:, :, 2:, :-2]-dAxdX[:, :, 2:, 2:])
      

        dSheardX = torch.abs(ShearStrain[:, :, :-2, :-2]-ShearStrain[:, :, 2:, 2:]+ShearStrain[:, :, 2:, :-2]-ShearStrain[:, :, 2:, 2:])
        dSheardY = torch.abs(dSheardX[:, :, :-2, :-2]-dSheardX[:, :, 2:, :-2]+dSheardX[:, :, :-2, 2:]-dSheardX[:, :, 2:, 2:])
 
        d = torch.abs(dAxdXX+dLatdYY-2.*dSheardY)
        flow_loss += i_weight *  d.mean()
 
      
    return flow_loss




def strainContinuityLoss(DispLat):

    dSRdY = ((DispLat[:, :, 1:, :] - DispLat[:, :, :-1, :]).abs())
    dSRdYY = ((dSRdY[:, :, 1:, :] - dSRdY[:, :, :-1, :]).abs())

    dSRdX = ((DispLat[:, :, :, 1:] - DispLat[:, :, :, :-1]).abs())
    dSRdXX = ((dSRdX[:, :, :, 1:] - dSRdX[:, :, :, :-1]).abs())
    
   # d = T.Resize(up_size)(dSRdY) +  T.Resize(up_size)(dSRdX)
    d = (torch.mean(dSRdYY) +  torch.mean(dSRdXX)) / 2


    return d
#%% check the data
for i, (input, target) in enumerate(val_loader):
        # strain = gaussian_filter(target, sigma=[3,3])
         #strain_GT=np.transpose(signal.convolve2d(np.transpose(strain), np.array([[1],[0],[-1]])))
         input
         target

Im1=input[0]
Im2=input[1]
Im1=Resize_transform(Im1).float()
Im2=Resize_transform(Im2).float()
target=Resize_transform(target).float()
Im1, Im2 = image_normalize(Im1, Im2)


img1_batch, img2_batch = preprocess(Im1, Im2)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")
        
j=1
plt.figure()
input1=input[j,1,:,:].squeeze()
plt.imshow(input1,aspect='auto',cmap='jet',vmin=0.00, vmax=1.00)
plt.colorbar()
plt.title('RF data')

#%% Pretrained model1

from torchvision.models.optical_flow import Raft_Large_Weights

def preprocess_raftLarge(img1_batch, img2_batch):
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms(img1_batch, img2_batch)

from torchvision.models.optical_flow import raft_large

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model1.eval()


#%%Define the model

args.gpus=[0]
torch.manual_seed(0)
lr = 0.00002
weight_decay=.00005
args.wdecay=.00005 
args.epsilon=1e-8
args.num_steps=10000000
args.clip=1.0
args.lr=lr
args.gpus=[0]
args.mixed_precision=True
args.batch_size=2

model = torch.nn.DataParallel(RAFT(args), device_ids=args.gpus).to(device)
for p in model.parameters():
         p.requires_grad =True 


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    
optimizer, scheduler = fetch_optimizer(args, model)  

scaler = GradScaler(enabled=args.mixed_precision)

# scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.5)

#%% Unsupervised  Training function

def train_epoch(model, model1,train_loader,optimizer,iters=12,flow_init=None):
    # Set train mode for both the encoder and the decoder
    model.train()
    model.module.freeze_bn()
    model1.eval()
    train_loss = 0.0
    valid=0.9
    gamma=0.85
    alpha=0.5
    epe_list=[]
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    
            
    for i, (input, target) in enumerate(train_loader):
         optimizer.zero_grad()
         #strain = gaussian_filter(target, sigma=[3,3])
         #strain_GT=np.transpose(signal.convolve2d(np.transpose(strain), np.array([[1],[0],[-1]])))
         Im1= Resize_transform(input[0])
         Im2=Resize_transform(input[1])
         Im1, Im2 = image_normalize(Im1, Im2)
         
         #Im1, Im2 = preprocess_raftLarge(Im1,Im2)
         Im1, Im2 = preprocess_raftLarge(Im1,Im2)
         #flo  = model1(Im1.to(device), Im2.to(device))[-1]
         _,flo = model1(Im1.to(device), Im2.to(device), iters=iters, flow_init=flo,test_mode=True)

         
         flo=Resize_init_flo(flo[0]).to(device)

         target =Resize_transform(torch.flip(target,[1]))
         #target =Resize_transform(target).to(device)   
         flow_predictions = model(Im1.to(device), Im2.to(device), iters=iters, flow_init=flo)
         epe = torch.sum((flow_predictions[-1].cpu() - target)**2, dim=0).sqrt()
         epe_list.append(epe.view(-1).numpy())

         loss_track = np.mean(np.concatenate(epe_list))
         
         #loss_track, metrics = sequence_loss(flow_predictions, target, valid,gamma)        
         print('loss_track: %f'%(loss_track.item()))
         loss_NCC =sequence_NCC (flow_predictions, Im1.to(device), Im2.to(device), valid,gamma)

         loss_smooth = sequence_gradNorm(flow_predictions, valid,gamma)
      
         loss_data = sequence_Dataloss(flow_predictions,  Im1.to(device), Im2.to(device), valid,gamma)

         loss = 2*loss_NCC + loss_smooth* alpha + 3*loss_data
        # loss = loss_NCC
         loss.backward()
         optimizer.step()

         train_loss+=loss.item()
         print('loss_single batch: %f' % (loss.item()/(args.batch_size)))
              
         torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
         

    scheduler.step()

    return train_loss / len(train_loader.dataset)  
 
   # return train_loss
#%%### ### Unsupervised Testing function

def test_epoch(model,model1,val_loader,iters=12,flow_init=None):

    model1.eval()
    model.eval()
    val_loss = 0.0
    epe_list = []

    with torch.no_grad(): # No need to track the gradients
        for i, (input, target) in enumerate(val_loader):

         Im1= Resize_transform(input[0])
         Im2=Resize_transform(input[1])
         target =Resize_transform(torch.flip(target,[1]))
         Im1, Im2 = image_normalize(Im1, Im2)
         Im1, Im2 = preprocess_raftLarge(Im1,Im2)

        
         _,flo = model1(Im1.to(device), Im2.to(device), iters=iters, flow_init=flo,test_mode=True)
         flo=Resize_init_flo(flo[0])
  
         target =Resize_transform(torch.flip(target,[1]))
         #target =Resize_transform(target).to(device) 
         _,flow_predictions = model(Im1.to(device), Im2.to(device), iters=iters, flow_init=flo,test_mode=True)


         epe = torch.sum((flow_predictions[0].cpu() - target)**2, dim=0).sqrt()
         epe_list.append(epe.view(-1).numpy())

         loss = np.mean(np.concatenate(epe_list))
        

         val_loss += loss.item()
    #return val_loss
    return val_loss / len(val_loader.dataset)
    #return epe
#%%Run the model
PATH = "SavedDir\state_dict_RAFTUSE_edited_ReUSELateral_strainCompat_LatReg_simulation.pt"
loss=1000.0
#best_model=model
num_epochs = 25
dim = (512 ,512)
for epoch in range(num_epochs):
   train_loss = train_epoch(model,model1,train_loader,optimizer,num_epoch=epoch)
   best_model,best_loss = test_epoch(model,model1,val_loader,loss,iters=12,flow_init=None)
  # print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   print('\n EPOCH {}/{} \t train loss {:.3f} '.format(epoch + 1, num_epochs,train_loss))


#%% test the pretrained RAFTUSNEt model 

PATH = "...\TrainedModel\state_dict_RAFTUSE.pt"
#torch.save(vae.state_dict(), PATH)
best_model = torch.nn.DataParallel(RAFT(args), device_ids=args.gpus).to(device)

best_model.load_state_dict(torch.load(PATH))
best_model.eval()

up_size=[2048, 512] #Upsample for test dataset 

iters=12


x_val_all=[]
with torch.no_grad(): # No need to track the gradients
         for i, (input, target) in enumerate(val_loader):
             
           
             
             Im1= Resize_transform(input[0])
             Im2=Resize_transform(input[1])
            
             
             Im1, Im2 = image_normalize(Im1, Im2)
             img1_batch, img2_batch = preprocess_raftLarge(Im1,Im2)
             
             flo  = model1(img1_batch.to(device), img2_batch.to(device))[-1]
             flo=Resize_init_flo(flo).to(device)
 
#             target =Resize_transform(torch.flip(target,[1]))
             #target =Resize_transform(target).to(device) 
             _,flow_predictions = best_model(img1_batch.to(device), img2_batch.to(device), iters=12, flow_init=None,test_mode=True)

 #             target = target.to(device)
             x_hat=flow_predictions.cpu().numpy()

#%% strain calculation
import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter
   
def Strain_Calc(ax_dis,sigma):
    ss=np.shape(ax_dis)

    strain = gaussian_filter(ax_dis, sigma=[sigma[0],sigma[1]],truncate=4)
    strain2=signal.convolve2d(strain, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),mode='same')
    # strain2=strain2[25:-25,5:-5]
    return strain2


def Strain_Calc_lateral(la_dis,sigma):
    ss=np.shape(la_dis)

    strain = gaussian_filter(la_dis, sigma=[sigma[0],sigma[1]],truncate=4)
    strain2=signal.convolve2d(strain, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),mode='same')
    # strain2=strain2[25:-25,5:-5]
    return strain2


#x_hat=flow_predictions.detach().cpu().numpy().astype(np.double)
x_hat=x_val_all[0:1]
ax_dis=x_hat[0,1,:,:]
lat_dis=x_hat[0,0,:,:]

strain_ax = Strain_Calc(ax_dis,sigma=[5,3])    
strain_la = Strain_Calc_lateral(lat_dis,sigma=[7,5])

plt.figure();plt.imshow(strain_ax,cmap='jet',aspect='auto',vmin=-8, vmax=0.13);plt.colorbar();plt.title('Axial Strain')
plt.figure();plt.imshow(strain_la,cmap='jet',aspect='auto',vmin=-0.05,vmax=1.5);plt.colorbar();plt.title('lateral Strain')
