import torch
import torch.nn.functional as F
import numpy as np
import math
#import scipy
def stack_all(list, dim=0):
    """Stack all iterables of torch tensors in a list (i.e. [[(tensor), (tensor)], [(tensor), (tensor)]])
    """
    return [torch.stack(s, dim) for s in list]

def get_patches(x, x_wind=143):
    kh, dh = (x_wind*2)+1, 1
    patches = x.unfold(2, kh, dh)
    patches = torch.squeeze(patches,dim=1).permute(0,1,3,2)
    return patches
def get_strain(disp, x_wind=143):
    d = x_wind*2+1
    Uxx_list = []
    disp = get_patches(disp,x_wind=x_wind)
    depthX = torch.linspace(1,d,d)
    depthX = torch.stack([depthX,torch.ones_like(depthX)]).float().permute(1,0).cuda()
    depthX = depthX.unsqueeze(0).repeat(disp.shape[1],1,1)
    XtX = depthX.permute(0,2,1).bmm(depthX)
    for i in range(len(disp)):
        # Cholesky decomposition
        XtY = depthX.permute(0,2,1).bmm(disp[i,...])
#        betas_cholesky, _ = torch.linalg.solve(XtY, XtX)
        betas_cholesky = torch.linalg.solve(XtX, XtY)
        Uxx = torch.squeeze(betas_cholesky[:,0,:])
        # pad to original size
        Uxx_list += [F.pad(Uxx, (0,0,x_wind, x_wind))]
    return torch.stack(Uxx_list).unsqueeze(1)


def GradNorm(disp):
    dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
    dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    dxx = torch.abs(dx[:, :, :, 1:] - dx[:, :,  :, :-1])
    dyy = torch.abs(dy[:, :, 1:, :] - dy[:, :,  :-1, :])
    d = (torch.mean(dxx) +  torch.mean(dyy)) / 2
    return d

def LNCC(pred, target, win = [160,8]):
    # Unfold pred and target into patches
    unfold = torch.nn.Unfold(kernel_size=(win[0],win[1]), stride = (win[0],win[1]))
    pred = unfold(pred).permute(0,2,1)
    target = unfold(target).permute(0,2,1)
    target_mean = torch.mean(target, dim=-1, keepdim=True)
    target_std = torch.std(target, dim=-1)
    pred_mean = torch.mean(pred, dim=-1, keepdim=True)
    pred_std = torch.std(pred, dim=-1)
    ncc = torch.sum((target - target_mean) * (pred - pred_mean),dim=-1).abs()  / (win[0]*win[1]*target_std*pred_std+1e-18)
    return torch.mean(ncc)
    

def NCC(pred, target):
    size_target_image = torch.numel(target)
    target_mean = torch.mean(target)
    target_std = torch.std(target)
    pred_mean = torch.mean(pred)
    pred_std = torch.std(pred)
    ncc = torch.sum((target -target_mean) * (pred - pred_mean)) / (size_target_image*target_std*pred_std)
    return ncc

def LNCC_voxelmorph(y_pred, y_true, win=[9, 9]):
    """
    Local (over window) normalized cross correlation loss.(voxelmorph pytorch)
    """
    I = y_true
    J = y_pred
    channel_size = I.size()[1]
    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(I.size())) - 2
    assert ndims in [
        1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # compute filters
    sum_filt = torch.ones([1, channel_size, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = (cross * cross) / (I_var * J_var + 1e-7)
    return torch.mean(cc) 

def RMSE(pred,target):
    target = -target
    pred = -pred
    return torch.sqrt(torch.mean((target-pred)**2))

def SNRe(strain):
    return torch.mean(strain) / torch.std(strain)

def NRMSE(pred,target):
    target = -target
    pred = -pred
    return (torch.sqrt(torch.mean((target-pred)**2))/torch.mean(target))*100
