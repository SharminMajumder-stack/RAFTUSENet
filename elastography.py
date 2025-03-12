# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:22:33 2024

@author: sharminjouty48
"""

"""
Created on Mon Feb  6 17:53:24 2023

@author: sharminjouty48
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
print(sys.executable)
import os
import os.path
#from scipy import *
import scipy.io as sio
from scipy.signal import hilbert
import copy
import numpy as np
#import cv2
#import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter
import glob
import torch
import torch.utils.data as data
import cv2
import random 
import torchvision.transforms as T
import torch.utils.data
#import scipy
#from torchvision.transforms import v2

#simData_size = [1968,236]
simData_size = [2048,256] ##for combined dataset
Resize_phantom = T.Compose([
       # T.Resize((984,236))
        T.Resize((simData_size))
    ]) 



def Data_form_phantom(Im1):
     s=np.shape(Im1)
     Data2=np.zeros([3,s[0],s[1]])
     a1=((hilbert_transform_imag(np.array(Im1,copy=True) )))
     #a1=((hilbert_transform_phase(np.array(Im1,copy=True) )))
     a2=np.array(Im1,copy=True)
     #a2=a2-np.min(a2)
     #a2=a2/np.max(a2)
     #a1=a1-np.min(a1)
     #a1=a1/np.max(a1)
     #a3=(hilbert_transform(np.array(a2,copy=True)))
     a3=(hilbert_transform(np.array(a2,copy=True)))
     
     #a3=a3-np.min(a3)
     #a3=a3/np.max(a3)
     
     # a1=hilbert_abs(Im1)
     # a1=a1-np.min(a1)
     # a1=a1/np.max(a1)
     # a2=np.array(Im1,copy=True)
     # a2=a2-np.min(a2)
     # a2=a2/np.max(a2)
     # a3=hilbert_phase(Im1)
     # a3=a3-np.min(a3)  
     # a3=a3/np.max(a3)
         
     Data2[0,:,:]=copy.deepcopy(a1)
     Data2[1,:,:]=copy.deepcopy(a2)
     Data2[2,:,:]=copy.deepcopy(a3)
     
     return Data2

def Data_form_Simu(Im1):
     s=np.shape(Im1)
     Data2=np.zeros([3,s[0],s[1]])
     a1=((hilbert_transform_imag(np.array(Im1,copy=True) )))
#     a1=((hilbert_transform_phase(np.array(Im1,copy=True) )))
     a2=np.array(Im1,copy=True)
    # a2=a2-np.min(a2)
     
     # if np.max(a2)==0:
     #     a2=a2/np.max(abs(a2))
         
     #a2=a2/np.max(a2)
     # a1=a1-np.min(a1)
     # if np.max(a1)==0:
     #    a1=a1/np.max(abs(a1))
     # a1=a1/np.max(a1)
     a3=(hilbert_transform(np.array(a2,copy=True)))

     #a3=a3-np.min(a3)
     #a3=a3/np.max(a3)
     
     # a1=hilbert_abs(Im1)
     # a1=a1-np.min(a1)
     # a1=a1/np.max(a1)
     # a2=np.array(Im1,copy=True)
     # a2=a2-np.min(a2)
     # a2=a2/np.max(a2)
     # a3=hilbert_phase(Im1)
     # a3=a3-np.min(a3)  
     # a3=a3/np.max(a3)
     # a1[math.isnan(a1)]=0
     # a2[math.isnan(a2)]=0
     # a3[math.isnan(a3)]=0
     # a1[math.isinf(a1)]=0
     # a2[math.isinf(a2)]=0
     # a3[math.isinf(a3)]=0
     Data2[0,:,:]=copy.deepcopy(a1)
     Data2[1,:,:]=copy.deepcopy(a2)
     Data2[2,:,:]=copy.deepcopy(a3)
     
     return Data2

def Data_form_sPIC(Im1):
     s=np.shape(Im1)
     Data2=np.zeros([3,s[0],s[1]])
     
     N=500.0
     a1=((hilbert_transform_imag(np.array(Im1,copy=True) )))
     a2=np.array(Im1,copy=True)

  
     a2=a2-np.min(a2)
     
     a2=a2/np.max(a2)
     

     a1=a1-np.min(a1)
     a1=a1/np.max(a1)
     a3=(hilbert_transform(np.array(a2,copy=True)))
        

     a3=a3-np.min(a3)
     
     a3=a3/np.max(a3)
     Data2[0,:,:]=copy.deepcopy(a1)
     Data2[1,:,:]=copy.deepcopy(a2)
     Data2[2,:,:]=copy.deepcopy(a3)
     return Data2
 

def Data_form_inVivo(Im1):
     s=np.shape(Im1)
     Data2=np.zeros([3,s[0],s[1]])
     a1=((amplitude_phase(np.array(Im1,copy=True) )))
    
     a2=np.array(Im1,copy=True)
    # a2=a2-np.min(a2)
     
     # if np.max(a2)==0:
     #     a2=a2/np.max(abs(a2))
         
     #a2=a2/np.max(a2)
     # a1=a1-np.min(a1)
     # if np.max(a1)==0:
     #    a1=a1/np.max(abs(a1))
     # a1=a1/np.max(a1)
     a3=(amplitude_envelope(np.array(a2,copy=True)))

     #a3=a3-np.min(a3)
     #a3=a3/np.max(a3)
     
     # a1=hilbert_abs(Im1)
     # a1=a1-np.min(a1)
     # a1=a1/np.max(a1)
     # a2=np.array(Im1,copy=True)
     # a2=a2-np.min(a2)
     # a2=a2/np.max(a2)
     # a3=hilbert_phase(Im1)
     # a3=a3-np.min(a3)  
     # a3=a3/np.max(a3)
     # a1[math.isnan(a1)]=0
     # a2[math.isnan(a2)]=0
     # a3[math.isnan(a3)]=0
     # a1[math.isinf(a1)]=0
     # a2[math.isinf(a2)]=0
     # a3[math.isinf(a3)]=0
     Data2[0,:,:]=copy.deepcopy(a1)
     Data2[1,:,:]=copy.deepcopy(a2)
     Data2[2,:,:]=copy.deepcopy(a3)
     
     return Data2 


   
def data_normalize(Data):
    s=np.shape(Data)
    z=np.zeros([s[0],s[1],s[2],s[3]],dtype=np.float32)  
    for i in range(s[0]):
        for j in range(s[1]):
            z[i,j,:,:]=(Data[i,j,:,:]-np.min(Data[i,j,:,:]))
            z[i,j,:,:]=z[i,j,:,:]/np.max(z[i,j,:,:])
    return z
 
def hilbert_transform(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
 #       h[:,i]=np.abs(hilbert(Im[:,i]-np.mean(Im[:,i])))
        h[:,i]=np.abs(hilbert(Im[:,i]))
    return h

def hilbert_transform_imag(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
#        mean_line=np.mean(Im[:,i])
#        h[:,i]=mean_line+np.imag(hilbert(Im[:,i]-mean_line))
        h[:,i]=np.imag(hilbert(Im[:,i]))
    return h

def hilbert_transform_phase(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
#        h[:,i]=np.angle(hilbert(Im[:,i]-np.mean(Im[:,i])))
        h[:,i]=np.angle(hilbert(Im[:,i]))
    return h

def add_gauss_blur(img):
    kernel_size_h=random.choice((1,3))
    kernel_size_v=random.choice((1,3))##7 added for the combined model, 5 added for supervised model
    blur = cv2.GaussianBlur(img,(kernel_size_v,kernel_size_h),0)

    return blur

def add_motion(img):
   kernel_size_h=random.choice((0,1,3,5))
   kernel_size_v=random.choice((1,3,5))
   #random.choice(((0,3),(0,5)))
# Create the vertical kernel.

   if  kernel_size_v !=0:
       kernel_v = np.zeros((kernel_size_v, kernel_size_v))
       kernel_v[:, int((kernel_size_v - 1)/2)] = np.ones(kernel_size_v) # Fill the middle row with ones.
       kernel_v /= kernel_size_v # Normalize.
       img = cv2.filter2D(img, -1, kernel_size_v)
  
# Create a copy of the same for creating the horizontal kernel.
# Apply the horizontal kernel.
   if  kernel_size_h !=0:
       kernel_h = np.zeros((kernel_size_h, kernel_size_h))
       kernel_h[int((kernel_size_h - 1)/2), :] = np.ones(kernel_size_h)
       kernel_h /= kernel_size_h
       img = cv2.filter2D(img, -1, kernel_size_h) 
  
   return img
              
#%% read simulation data

def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]

    return train_samples, test_samples


def default_loader(root, inputs,target):
    
    rf1=sio.loadmat(inputs[0])
    rf2=sio.loadmat(inputs[1])
    rf1=np.array(rf1['rf'])
    rf2=np.array(rf2['rf'])
    #Adding motion
    # if random.randint(0, 1)==1:
    # #     rf2=add_motion(rf2)
    #    rf1=add_gauss_blur(rf1)
       
    # if random.randint(0, 1)==1:
    # #     rf2=add_motion(rf2)
    #    rf2=add_gauss_blur(rf2)
    # rf1=Data_form_Rivaz(rf1[40:-40,10:-10]).astype(np.float32)
    # rf2=Data_form_Rivaz(rf2[40:-40,10:-10]).astype(np.float32)
    maxIm = rf1.max()
    rf1 = rf1/maxIm
    rf2 = rf2/maxIm
    rf1=Data_form_Simu(rf1).astype(np.float32)
    rf2=Data_form_Simu(rf2).astype(np.float32)
    # rf1[np.isnan(rf1)]=0
    # rf2[np.isnan(rf2)]=0
    # rf1=Data_form_complex(rf1[40:-40,10:-10]).astype(np.float32)
    # rf2=Data_form_complex(rf2[40:-40,10:-10]).astype(np.float32)    
    #rf1=Data_form(rf1).astype(np.float32)
    #rf2=Data_form(rf2).astype(np.float32)
    rf1=torch.from_numpy(rf1.copy())
    rf2=torch.from_numpy(rf2.copy())

    Disp_GT=sio.loadmat(target)
    Disp_GT=Disp_GT['DispGT'].astype(np.float32)  
    Disp_GT=torch.from_numpy(np.array(Disp_GT).copy())
   # Disp_GT=torch.from_numpy(np.array(Disp_GT[:,40:-40,10:-10]).copy())
    rf=[rf1,rf2]
  #  return [rf1,rf2, Disp_GT]
    return [rf, Disp_GT]
#    return [sio.loadmat(inp) for inp in inps], sio.loadmat(disp)

#    return [sio.loadmat(inp).astype(np.float32) for inp in inps], [sio.loadmat(disp).astype(np.float32) for disp in disps]
def default_loader_sPICTURE(root, inputs,target):
    
    rf1=sio.loadmat(inputs[0])
    rf2=sio.loadmat(inputs[1])
    rf1=np.array(rf1['rf'])
    rf2=np.array(rf2['rf'])
    #Adding motion
    # if random.randint(0, 1)==1:
    # #     rf2=add_motion(rf2)
    #    rf1=add_gauss_blur(rf1)
       
    # if random.randint(0, 1)==1:
    # #     rf2=add_motion(rf2)
    #    rf2=add_gauss_blur(rf2)

    rf1=Data_form_sPIC(rf1).astype(np.float32)
    rf2=Data_form_sPIC(rf2).astype(np.float32)
    # rf1[np.isnan(rf1)]=0
    # rf2[np.isnan(rf2)]=0
    # rf1=Data_form_complex(rf1[40:-40,10:-10]).astype(np.float32)
    # rf2=Data_form_complex(rf2[40:-40,10:-10]).astype(np.float32)    
    #rf1=Data_form(rf1).astype(np.float32)
    #rf2=Data_form(rf2).astype(np.float32)
    rf1=torch.from_numpy(rf1.copy())
    rf2=torch.from_numpy(rf2.copy())

    Disp_GT=sio.loadmat(target)
    Disp_GT=Disp_GT['DispGT'].astype(np.float32)  
    Disp_GT=torch.from_numpy(np.array(Disp_GT).copy())
   # Disp_GT=torch.from_numpy(np.array(Disp_GT[:,40:-40,10:-10]).copy())
    rf=[rf1,rf2]
  #  return [rf1,rf2, Disp_GT]
    return [rf, Disp_GT]

def default_RFData(root, inputs,target):
    
    rf1=sio.loadmat(inputs[0])
    rf2=sio.loadmat(inputs[1])
    rf1=np.array(rf1['rf'])
    rf2=np.array(rf2['rf'])

    rf1=torch.from_numpy(rf1.copy())
    rf2=torch.from_numpy(rf2.copy())

    Disp_GT=sio.loadmat(target)
    Disp_GT=Disp_GT['DispGT'].astype(np.float32)  
    Disp_GT=torch.from_numpy(np.array(Disp_GT).copy())
   # Disp_GT=torch.from_numpy(np.array(Disp_GT[:,40:-40,10:-10]).copy())
    rf=[rf1,rf2]
  #  return [rf1,rf2, Disp_GT]
    return [rf, Disp_GT] 

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
       # print(index, "INDEX")
        inputs, target = self.path_list[index]

        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)

class ListDataset_sPICTURE(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader_sPICTURE):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)

class ListDataset_RFData(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_RFData):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)
    
def make_dataset(dataset_dir, split):
    dis_dir=r"D:\OneDrive - Texas A&M University\TAMU\Research\ML_DL_AI\Elastography\Data\Sharmin\MyData\Label"
    assert(os.path.isdir(os.path.join(dis_dir)))
    inp_dir="Input\RF"
    assert(os.path.isdir(os.path.join(dataset_dir,inp_dir)))
    
    images = []
 
    for dis_gt_dir in sorted(glob.glob(os.path.join(dis_dir ,'*'))):
    
#    dis_file_dir = os.path.relpath(dis_map,os.path.join(dataset_dir,dis_dir,dis_file_dir))
        model_dir, model_num = os.path.split(dis_gt_dir)
   # no_ext_filename = os.path.splitext(filename)[0]
    #prefix, frame_nb = no_ext_filename.split('_')
    
    #frame_nb = int(frame_nb)
    
        rf_dir=os.path.join(dataset_dir,inp_dir, model_num)
        for sample_num in sorted(glob.glob(os.path.join(rf_dir,'*'))):
   #     rf_map=os.path.join(dataset_dir,img_dir, dis_file_dir,sample_num)
           for rf_frame in sorted(glob.glob(os.path.join(sample_num,'*.mat'))):
            
#            dis_map = os.path.relpath(dis_map,os.path.join(dataset_dir,dis_dir))
            rf_dir1, frame_name = os.path.split(rf_frame)
            no_ext_filename = os.path.splitext(frame_name)[0]
            prefix, frame_num ,CenterFreq = no_ext_filename.split('_')
            frame_num = int(frame_num)
            

            if frame_num>1:
               rf1 = os.path.join(rf_dir1, '{}_{}_{}.mat'.format(prefix,'1',CenterFreq))
               rf2=rf_frame
               rf=[rf1,rf2]
               dis_map_gt = os.path.join(dis_gt_dir,'{}_{}.mat'.format(prefix, frame_num))
          
               images.append([rf,dis_map_gt])
               #images.append([rf1,rf2,dis_map_gt])
   # train_samples=images[0:int(0.99*len(images))]
    #test_samples=images[int(0.99*len(images)):]
    return split2list(images, split, default_split=0.85)
    #return train_samples,test_samples
def make_TrainTestDataset(dataset_dir, split):
    dis_dir=r"D:\OneDrive - Texas A&M University\TAMU\Research\ML_DL_AI\Elastography\Data\Sharmin\MyData\Label"
    assert(os.path.isdir(os.path.join(dis_dir)))

    Train_dir="Input\RF"
    assert(os.path.isdir(os.path.join(dataset_dir,Train_dir))) 

    Test_dir="Input\TestData_5MHz"
    assert(os.path.isdir(os.path.join(dataset_dir,Test_dir)))  

    train_samples,test_samples = [],[]
    for dis_gt_dir in sorted(glob.glob(os.path.join(dis_dir ,'*'))):          
        model_dir, model_num = os.path.split(dis_gt_dir)
        rf_Train_dir=os.path.join(dataset_dir,Train_dir, model_num)
        rf_Test_dir=os.path.join(dataset_dir,Test_dir, model_num)        
        
        for sample_num in sorted(glob.glob(os.path.join(rf_Train_dir,'*'))):
          for rf_frame in sorted(glob.glob(os.path.join(sample_num,'*.mat'))):
           
            rf_dir1, frame_name = os.path.split(rf_frame)
            no_ext_filename = os.path.splitext(frame_name)[0]
            prefix, frame_num ,CenterFreq = no_ext_filename.split('_')
            frame_num = int(frame_num)           

            if frame_num>1:
               rf1 = os.path.join(rf_dir1, '{}_{}_{}.mat'.format(prefix,'1',CenterFreq))
               rf2=rf_frame
               rf=[rf1,rf2]
               dis_map_gt = os.path.join(dis_gt_dir,'{}_{}.mat'.format(prefix, frame_num))
               train_samples.append([rf,dis_map_gt])
               #images.append([rf1,rf2,dis_map_gt]) 
 
        for sample_num in sorted(glob.glob(os.path.join(rf_Test_dir,'*'))):
          for rf_frame in sorted(glob.glob(os.path.join(sample_num,'*.mat'))):
            
            rf_dir1, frame_name = os.path.split(rf_frame)
            no_ext_filename = os.path.splitext(frame_name)[0]
            prefix, frame_num ,CenterFreq = no_ext_filename.split('_')
            frame_num = int(frame_num)
            
            if frame_num>1:
               rf1 = os.path.join(rf_dir1, '{}_{}_{}.mat'.format(prefix,'1',CenterFreq))
               rf2=rf_frame
               rf=[rf1,rf2]
               dis_map_gt = os.path.join(dis_gt_dir,'{}_{}.mat'.format(prefix, frame_num))
               test_samples.append([rf,dis_map_gt])   
#    return split2list(images, split, default_split=0.85)
    return train_samples,test_samples

def elastography_data(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform,co_transform )

    return train_dataset, test_dataset

def elastography_TrainTestData(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_TrainTestDataset(root, split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform,co_transform )

    return train_dataset, test_dataset

def elastography_ActualRF_InpData(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_TrainTestDataset(root, split)
    train_dataset = ListDataset_RFData(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset_RFData(root, test_list, transform, target_transform,co_transform )

    return train_dataset, test_dataset

def elastography_TrainTestData_SPIC(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_TrainTestDataset(root, split)
    train_dataset = ListDataset_sPICTURE(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset_sPICTURE(root, test_list, transform, target_transform,co_transform )

    return train_dataset, test_dataset
#%% Phantom data read

def default_loader_phantom(root, inputs,target):
    
    rf=sio.loadmat(inputs)
    rf=rf['Data']
    rf1=np.array(rf[0])
    rf2=np.array(rf[1])
    maxIm = rf1.max()
    rf1 = rf1/maxIm
    rf2 = rf2/maxIm
    #Adding motion
    
    # if random.randint(0, 1)==1:
    #     rf2=add_motion(rf2)
       
    # rf1=Data_form_phantom(rf1[40:-40,5:-4]).astype(np.float32)
    # rf2=Data_form_phantom(rf2[40:-40,5:-4]).astype(np.float32)
    rf1=Data_form_phantom(rf1).astype(np.float32)
    rf2=Data_form_phantom(rf2).astype(np.float32)
    
    # rf1=Data_form_complex(rf1[40:-40,10:-10]).astype(np.float32)
    # rf2=Data_form_complex(rf2[40:-40,10:-10]).astype(np.float32)    
    #rf1=Data_form(rf1).astype(np.float32)
    #rf2=Data_form(rf2).astype(np.float32)
    rf1=torch.from_numpy(rf1.copy())
    rf2=torch.from_numpy(rf2.copy())
    rf1=Resize_phantom(rf1)
    rf2=Resize_phantom(rf2)
    rfData=[rf1,rf2]
 #   Disp_GT=torch.from_numpy(np.array(rf[:,40:-40,10:-10].astype(np.float32)))#to match the simulated data for the code. otherwise invalid
 
    Disp_GT=torch.from_numpy(np.array(rf.astype(np.float32)))#to match the simulated data for the code. otherwise invalid
    Disp_GT=Resize_phantom(Disp_GT)
  #  return [rf1,rf2, Disp_GT]
    return [rfData, Disp_GT]
#    return [sio.loadmat(inp) for inp in inps], sio.loadmat(disp)

#    return [sio.loadmat(inp).astype(np.float32) for inp in inps], [sio.loadmat(disp).astype(np.float32) for disp in disps]
def default_loader_phantom_Aug(root, inputs,target):
    
    rf=sio.loadmat(inputs)
    rf=rf['Data']
    rf1=np.array(rf[0])
    rf2=np.array(rf[1])
    # rf2=add_gauss_blur(rf2)
    
    maxIm = rf1.max()
    rf1 = rf1/maxIm
    rf2 = rf2/maxIm
    #Adding motion
    
    # if random.randint(0, 1)==1:
    #     rf2=add_motion(rf2)
       
    # rf1=Data_form_phantom(rf1[40:-40,5:-4]).astype(np.float32)
    # rf2=Data_form_phantom(rf2[40:-40,5:-4]).astype(np.float32)
    rf1=Data_form_phantom(rf1).astype(np.float32)
    rf2=Data_form_phantom(rf2).astype(np.float32)
    
    # rf1=Data_form_complex(rf1[40:-40,10:-10]).astype(np.float32)
    # rf2=Data_form_complex(rf2[40:-40,10:-10]).astype(np.float32)    
    #rf1=Data_form(rf1).astype(np.float32)
    #rf2=Data_form(rf2).astype(np.float32)
    rf1=torch.from_numpy(rf1.copy())
    rf2=torch.from_numpy(rf2.copy())
    rf1=Resize_phantom(rf1)
    rf2=Resize_phantom(rf2)
    rfData=[rf1,rf2]
 #   Disp_GT=torch.from_numpy(np.array(rf[:,40:-40,10:-10].astype(np.float32)))#to match the simulated data for the code. otherwise invalid
 
    Disp_GT=torch.from_numpy(np.array(rf.astype(np.float32)))#to match the simulated data for the code. otherwise invalid
    Disp_GT=Resize_phantom(Disp_GT)
  #  return [rf1,rf2, Disp_GT]
    return [rfData, Disp_GT]

class ListDataset_phantom(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=None):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)
    
def make_dataset_phantom(dataset_dir, split):
    rf_dir=r"D:\OneDrive - Texas A&M University\TAMU\Research\DeepLearning\Elastography\Data\Rivaz_group\TUFFC_2022_Bi_Directional_elasto\CIRS phantom (model 059)"
    assert(os.path.isdir(os.path.join(rf_dir)))
        
    images = []

    for rf in sorted(glob.glob(os.path.join(rf_dir ,'*.mat'))):
       rf=rf
       target=rf
       images.append([rf,target])
               #images.append([rf1,rf2,dis_map_gt])
   # train_samples=images[0:int(0.99*len(images))]
    #test_samples=images[int(0.99*len(images)):]
    return split2list(images, split, default_split=0.85)

def elastography_data_phantom(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset_phantom(root, split)
    train_dataset = ListDataset_phantom(root, train_list, transform, target_transform, co_transform,loader=default_loader_phantom)
    test_dataset = ListDataset_phantom(root, test_list, transform, target_transform,co_transform,loader=default_loader_phantom )

#    train_dataset2 = ListDataset_phantom(root, train_list, transform, target_transform, co_transform,default_loader_phantom_Aug)
#    test_dataset2 = ListDataset_phantom(root, test_list, transform, target_transform,co_transform,default_loader_phantom_Aug )


#    train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
#    test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])
    return train_dataset, test_dataset
#%%
up_size = [2048,256]
def up_sample_Data_sPICTURE(Data,si):
    s=np.shape(Data)
    W=si[1]
    H=si[0]
    res0_test=np.zeros([s[0],H,W],dtype=np.float32)
    for i in range(s[0]):
       
            res0_test[i,:,:] = cv2.resize(Data[i,:,:], dsize=(W, H), interpolation=cv2.INTER_CUBIC)
         
    return res0_test

def hilbert_transform_sPICTURE(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
        h[:,i]=np.abs(hilbert(Im[:,i]-np.mean(Im[:,i])))
    return h

def hilbert_transform_imag_sPICTURE(Im):
    s=np.shape(Im)
    h=np.zeros([s[0],s[1]],dtype=np.float32)
    for i in range(s[1]):
        mean_line=np.mean(Im[:,i])
        h[:,i]=mean_line+np.imag(hilbert(Im[:,i]-mean_line))
    return h

def data_normalize_sPICTURE(Data):
    s=np.shape(Data)
    z=np.zeros([s[0],s[1],s[2]],dtype=np.float32)  
    for i in range(s[0]):
            z[i,:,:]=(Data[i,:,:]-np.min(Data[i,:,:]))
            z[i,:,:]=z[i,:,:]/np.max(z[i,:,:])
    return z

def Data_form_sPICTURE(Im1):
     s=np.shape(Im1)
     Data2=np.zeros([3,s[0],s[1]])
     
     N=500.0
     a1=((hilbert_transform_imag_sPICTURE(np.array(Im1,copy=True) )))
     a2=np.array(Im1,copy=True)

  
     a2=a2-np.min(a2)
     
     a2=a2/np.max(a2)
     

 

     a1=a1-np.min(a1)
     a1=a1/np.max(a1)
     a3=(hilbert_transform_sPICTURE(np.array(a2,copy=True)))
     

     a3=a3-np.min(a3)
     
     a3=a3/np.max(a3)
     Data2[0,:,:]=copy.deepcopy(a1)
     Data2[1,:,:]=copy.deepcopy(a2)
     Data2[2,:,:]=copy.deepcopy(a3)
     return Data2
def default_loader_sPICTURE(root, inputs,target):
    
    rf1=sio.loadmat(inputs[0])
    rf2=sio.loadmat(inputs[1])
    rf1=np.array(rf1['rf'])
    rf2=np.array(rf2['rf'])
    #Adding motion
    
    # if random.randint(0, 1)==1:
    #     rf2=add_motion(rf2)
       
    rf1=up_sample_Data_sPICTURE(data_normalize_sPICTURE(Data_form_sPICTURE(rf1[40:-40,5:-4])),up_size).astype(np.float32)
    rf2=up_sample_Data_sPICTURE(data_normalize_sPICTURE(Data_form_sPICTURE(rf2[40:-40,5:-4])),up_size).astype(np.float32)

    # rf1[np.isnan(rf1)]=0
    # rf2[np.isnan(rf2)]=0
    # rf1=Data_form_complex(rf1[40:-40,10:-10]).astype(np.float32)
    # rf2=Data_form_complex(rf2[40:-40,10:-10]).astype(np.float32)    
    #rf1=Data_form(rf1).astype(np.float32)
    #rf2=Data_form(rf2).astype(np.float32)
    rf1=torch.from_numpy(rf1.copy())
    rf2=torch.from_numpy(rf2.copy())

    Disp_GT=sio.loadmat(target)
    Disp_GT=Disp_GT['DispGT'].astype(np.float32)  

    Disp_GT=up_sample_Data_sPICTURE(Disp_GT[:,40:-40,5:-4],up_size).astype(np.float32)
    Disp_GT=torch.from_numpy(Disp_GT.copy())

    rf=[rf1,rf2]
  #  return [rf1,rf2, Disp_GT]
    return [rf, Disp_GT]



def elastography_data_sPICTURE(root, transform=None, target_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset_sPICTURE(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset_sPICTURE(root, test_list, transform, target_transform,co_transform )

    return train_dataset, test_dataset

class ListDataset_sPICTURE(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader_sPICTURE):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)