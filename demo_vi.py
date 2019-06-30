import os, sys
import os.path
import torch
import numpy as np
import cv2
from scipy.misc import imsave

from torch.utils import data
from davis import DAVIS

from model import generate_model
from utils import *
import time

import pims
import subprocess as sp
import pickle
import pdb

class Object():
    pass
opt = Object()
opt.crop_size = 256
########## DAVIS
DAVIS_ROOT = './DAVIS_demo'
DTset = DAVIS(DAVIS_ROOT, imset='2016/demo_davis.txt', size=(opt.crop_size, opt.crop_size))
DTloader = data.DataLoader(DTset, batch_size=1, shuffle=False, num_workers=1)

opt.search_range = 4 # fixed as 4: search range for flow subnetworks
opt.pretrain_path = 'results/vinet_agg_rec/save_agg_rec.pth'
opt.result_path = 'results/vinet_agg_rec'

opt.model = 'vinet_final'
opt.batch_norm = False
opt.no_cuda = False # use GPU
opt.no_train = True
opt.test = True
opt.t_stride = 3
opt.loss_on_raw = False
opt.prev_warp = True
opt.save_image = False
opt.save_video = True


def createVideoClip(clip, folder, name, size=[256,256]):

    vf = clip.shape[0]
    command = [ 'ffmpeg',
    '-y',  # overwrite output file if it exists
    '-f', 'rawvideo',
    '-s', '%dx%d'%(size[1],size[0]), #'256x256', # size of one frame
    '-pix_fmt', 'rgb24',
    '-r', '15', # frames per second
    '-an',  # Tells FFMPEG not to expect any audio
    '-i', '-',  # The input comes from a pipe
    '-vcodec', 'libx264',
    '-b:v', '1500k',
    '-vframes', str(vf), # 5*25
    '-s', '%dx%d'%(size[1],size[0]), #'256x256', # size of one frame
    folder+'/'+name ]
    #sfolder+'/'+name 
    pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    print(err)

def to_img(x):
    tmp = (x[0,:,0,:,:].cpu().data.numpy().transpose((1,2,0))+1)/2
    tmp = np.clip(tmp,0,1)*255.
    return tmp.astype(np.uint8)

model, _ = generate_model(opt)
print('Number of model parameters: {}'.format( sum([p.data.nelement() for p in model.parameters()])))

model.eval()
ts = opt.t_stride
folder_name = 'davis_256'
pre = 20

with torch.no_grad():
    for seq, (inputs, masks, info) in enumerate(DTloader):

        idx = torch.LongTensor([i for i in range(pre-1,-1,-1)])
        pre_inputs = inputs[:,:,:pre].index_select(2,idx)
        pre_masks = masks[:,:,:pre].index_select(2,idx)
        inputs = torch.cat((pre_inputs, inputs),2)
        masks = torch.cat((pre_masks, masks),2)

        bs = inputs.size(0)
        num_frames = inputs.size(2)
        seq_name = info['name'][0]

        save_path = os.path.join(opt.result_path, folder_name, seq_name)
        if not os.path.exists(save_path) and opt.save_image:
            os.makedirs(save_path)

        inputs = 2.*inputs - 1
        inverse_masks = 1-masks
        masked_inputs = inputs.clone()*inverse_masks

        masks = to_var(masks)
        masked_inputs = to_var(masked_inputs)
        inputs = to_var(inputs)

        total_time = 0.
        in_frames = []
        out_frames = []

        lstm_state = None

        for t in range(num_frames):
            masked_inputs_ = []
            masks_ = []        

            if t < 2*ts:
                masked_inputs_.append(masked_inputs[0,:,abs(t-2*ts)])
                masked_inputs_.append(masked_inputs[0,:,abs(t-1*ts)])
                masked_inputs_.append(masked_inputs[0,:,t])
                masked_inputs_.append(masked_inputs[0,:,t+1*ts])
                masked_inputs_.append(masked_inputs[0,:,t+2*ts])
                masks_.append(masks[0,:,abs(t-2*ts)])
                masks_.append(masks[0,:,abs(t-1*ts)])
                masks_.append(masks[0,:,t])
                masks_.append(masks[0,:,t+1*ts])
                masks_.append(masks[0,:,t+2*ts])
            elif t > num_frames-2*ts-1:
                masked_inputs_.append(masked_inputs[0,:,t-2*ts])
                masked_inputs_.append(masked_inputs[0,:,t-1*ts])
                masked_inputs_.append(masked_inputs[0,:,t])
                masked_inputs_.append(masked_inputs[0,:,-1 -abs(num_frames-1-t - 1*ts)])
                masked_inputs_.append(masked_inputs[0,:,-1 -abs(num_frames-1-t - 2*ts)])
                masks_.append(masks[0,:,t-2*ts])
                masks_.append(masks[0,:,t-1*ts])
                masks_.append(masks[0,:,t])
                masks_.append(masks[0,:,-1 -abs(num_frames-1-t - 1*ts)])
                masks_.append(masks[0,:,-1 -abs(num_frames-1-t - 2*ts)])   
            else:
                masked_inputs_.append(masked_inputs[0,:,t-2*ts])
                masked_inputs_.append(masked_inputs[0,:,t-1*ts])
                masked_inputs_.append(masked_inputs[0,:,t])
                masked_inputs_.append(masked_inputs[0,:,t+1*ts])
                masked_inputs_.append(masked_inputs[0,:,t+2*ts])
                masks_.append(masks[0,:,t-2*ts])
                masks_.append(masks[0,:,t-1*ts])
                masks_.append(masks[0,:,t])
                masks_.append(masks[0,:,t+1*ts])
                masks_.append(masks[0,:,t+2*ts])            

            masked_inputs_ = torch.stack(masked_inputs_).permute(1,0,2,3).unsqueeze(0)
            masks_ = torch.stack(masks_).permute(1,0,2,3).unsqueeze(0)

            start = time.time()
            prev_mask = masks_[:,:,2] if t==0 else to_var(torch.zeros(masks_[:,:,2].size()))
            prev_ones = to_var(torch.ones(prev_mask.size()))
            prev_feed = torch.cat([masked_inputs_[:,:,2,:,:],prev_ones, prev_ones*prev_mask], dim=1) if t==0 else torch.cat([outputs.detach().squeeze(2), prev_ones, prev_ones*prev_mask], dim=1)

            #outputs, flows, lstm_state, occ_masks, flow_256 = model(masked_inputs_, masks_, lstm_state, prev_feed, t)
            outputs, _, _, _, _ = model(masked_inputs_, masks_, lstm_state, prev_feed, t)

            lstm_state = None
            end = time.time() - start
            if lstm_state is not None:
                lstm_state = repackage_hidden(lstm_state)

            total_time += end
            if t>pre:
                print('{}th frame of {} is being processed'.format(t-pre, seq_name))
                out_frame = to_img(outputs)  
                if opt.save_image:            
                    cv2.imwrite(os.path.join(save_path,'%05d.png'%(t)), out_frame)
                out_frames.append(out_frame[:,:,::-1])

        if opt.save_video:
            final_clip = np.stack(out_frames)
            video_path = os.path.join(opt.result_path, folder_name)
            if not os.path.exists(video_path):
                os.makedirs(video_path)

            createVideoClip(final_clip, video_path, '%s.mp4'%(seq_name), [opt.crop_size, opt.crop_size])
            print('Predicted video clip {} saving'.format(folder_name))   

