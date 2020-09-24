"""
It is not yet cleaned up to be runnable in conjunction, 
but to provide a reference algorithm. 

For easier training, three-stage learning is recommended.
Stage-1 & 2: w.o recurrence 
(w_ST, w_LT, w_Flow = 0, 0, 0,
t_stride, sample_duration, sample_frames = 3, 13, 1):
1: temporal aggregation with 1 support and 1 target frame
2: temporal aggregation with 4 support and 1 target frame
Stage-3: with recurrence (w_ST, w_LT, w_Flow = 1, 1, 10,
t_stride, sample_duration, sample_frames = 3, 16, 4):
3: train with recurrence and short- and long-term temporal loss
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pytorch_ssim
from pytorch_misc import clip_grad_norm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from networks.resample2d_package.modules.resample2d import Resample2d
from networks.resample2d_package.resample2d import Resample2d
import networks

class Object(object):
    pass
""" Flownet """
args = Object()
args.rgb_max = 1.0
args.fp16 = False
FlowNet = networks.FlowNet2(args, requires_grad=False)
model_filename = os.path.join("pretrained_models", "FlowNet2_checkpoint.pth.tar")
checkpoint = torch.load(model_filename)
FlowNet.load_state_dict(checkpoint['state_dict'])
FlowNet = FlowNet.cuda()
""" Submodules """
flow_warping = Resample2d().cuda()
downsampler = nn.AvgPool2d((2, 2), stride=2).cuda()

def norm(t):
    return torch.sum(t*t, dim=1, keepdim=True) 

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train_lstm_epoch(epoch, data_loader, model, criterion_L1, criterion_ssim, optimizer, opt):

    opt.w_ST, opt.w_LT, opt.w_Flow = 1.0, 1.0, 10.0
    model.train()
    ts = opt.t_stride 
    
    ### start epoch
    for i, (inputs, masks, _) in enumerate(data_loader):
        # inputs: BxCxTxHxW
        bs = inputs.size(0)
        midx = (inputs.size(2) - 1)//2
        inputs = 2.*inputs - 1. # [-1 1] 
        inverse_masks = 1.-masks
        masked_inputs = inputs.clone()*inverse_masks

        frame_i, frame_mi, frame_m = [], [], []
        for tt in range(opt.sample_frames):
            slices = [x*ts+tt for x in range(5)]
            frame_i.append(to_var(inputs[:,:,midx*ts+tt,:,:])) 
            frame_mi.append(to_var(masked_inputs[:,:,slices,:,:]))
            frame_m.append(to_var(masks[:,:,slices,:,:]))
       
        optimizer.zero_grad()
        lstm_state = None
        ST_loss, LT_loss = 0, 0
        RECON_loss, HOLE_loss = 0, 0
        flow_loss = 0

        ### forward
        prev_mask = frame_m[0][:,:,midx,:,:]
        prev_ones = to_var(torch.ones(prev_mask.size()))
        prev_feed = torch.cat([frame_mi[0][:,:,midx,:,:],prev_ones, prev_ones*prev_mask], dim=1)
        frame_o1, _, lstm_state, _ ,occs = model(frame_mi[0], frame_m[0], lstm_state, prev_feed)
        # Turned out it still works w.o.LSTM. May work with lstm_state = None 
        lstm_state = None if opt.no_lstm else repackage_hidden(lstm_state)
        frame_o1 = frame_o1.squeeze(2)
        
        RECON_loss += 1*criterion_L1(frame_o1, frame_i[0]) -
                      criterion_ssim(frame_o1, frame_i[0])
        HOLE_loss += 5*criterion_L1(
            frame_o1*frame_m[0][:,:,midx,:,:].expand_as(frame_o1), 
            frame_i[0]*frame_m[0][:,:,midx,:,:].expand_as(frame_o1)
            )
        frame_o = []
        frame_o.append(frame_o1)

        ### if opt.sample_frames > 1 , Recurrence learning
        for tt in range(1, opt.sample_frames):

            frame_i1, frame_m1 = frame_i[tt-1], frame_m[tt-1]
            frame_mi2 = frame_mi[tt]
            frame_i2, frame_m2 = frame_i[tt], frame_m[tt]
            frame_o1 = frame_o1.detach() if tt == 1 else frame_o2.detach()

            prev_mask = to_var(torch.zeros(frame_m2[:,:,midx,:,:].size()))
            prev_ones = to_var(torch.ones(prev_mask.size()))
            prev_feed = torch.cat([frame_o1,prev_ones, prev_ones*prev_mask], dim=1)
            frame_o2, _, lstm_state, _, occs, flow6_256 = model(
                frame_mi2, frame_m2, lstm_state, prev_feed, None, 1)
            if opt.loss_on_raw:
                frame_o2_raw = frame_o2[1].squeeze(2)
                frame_o2 = frame_o2[0]
            frame_o2 = frame_o2.squeeze(2)
            ### detach from graph and avoid memory accumulation
            lstm_state = None if opt.no_lstm else repackage_hidden(lstm_state)

            frame_o.append(frame_o2)
            RECON_loss += criterion_L1(frame_o2, frame_i2) - 
                          criterion_ssim(frame_o2, frame_i2)
            HOLE_loss += 5*criterion_L1(
                frame_o2*frame_m2[:,:,midx,:,:].expand_as(frame_o2), 
                frame_i2*frame_m2[:,:,midx,:,:].expand_as(frame_i2)
                    )

            if opt.loss_on_raw:
                RECON_loss += criterion_L1(frame_o2_raw, frame_i2) - 
                              criterion_ssim(frame_o2_raw, frame_i2)
                HOLE_loss += 5*criterion_L1(
                    frame_o2_raw*frame_m2[:,:,midx,:,:].expand_as(frame_o2_raw),
                    frame_i2*frame_m2[:,:,midx,:,:].expand_as(frame_i2))

            ### short-term temporal loss
            if opt.w_ST > 0:
                flow_i21 = FlowNet(frame_i2, frame_i1)
                warp_i1 = flow_warping(frame_i1, flow_i21)
                warp_o1 = flow_warping(frame_o1, flow_i21)
                noc_mask2 = torch.exp( -50. * torch.sum(
                    frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)
                ST_loss += criterion_L1(
                    frame_o2 * noc_mask2, warp_i1 * noc_mask2)
            
                conf = (norm(frame_i2 - warp_i1) < 0.02).float()
                flow_loss = criterion_L1(flow6_256 * conf, flow_i21 * conf)
                warp_i1_ = flow_warping(frame_i1, flow6_256)
                flow_loss += criterion_L1(warp_i1_ * conf, frame_i2 * conf)
                warp_o1_ = flow_warping(frame_o1, flow6_256)
                flow_loss += criterion_L1(
                    frame_o2 * conf, warp_o1_.detach() * conf)
                    
        if opt.w_LT > 0:
            t1 = 0    
            for t2 in range(t1 + 2, opt.sample_frames):
                frame_i1, frame_i2 = frame_i[t1], frame_i[t2]
                frame_o1 = frame_o[t1].detach()
                frame_o1.requires_grad = False
                frame_o2 = frame_o[t2]

                flow_i21 = FlowNet(frame_i2, frame_i1)
                warp_i1 = flow_warping(frame_i1, flow_i21)
                warp_o1 = flow_warping(frame_o1, flow_i21)
                noc_mask2 = torch.exp( -50. * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)
                LT_loss += criterion_L1(frame_o2 * noc_mask2, warp_i1 * noc_mask2)

        overall_loss = RECON_loss + GRAD_loss + HOLE_loss + opt.w_ST*ST_loss + opt.w_LT*LT_loss + opt.w_FLOW*flow_loss

        overall_loss.backward()
        optimizer.step()

    return overall_loss.data[0]

    