import os, os.path
import cv2
from PIL import Image
import numpy as np

# vid_name = 'bmx-trees'
# mode = 'horizontal' # OR 'vertical'
ratio = 0.5
H, W = 512, 512
bg_root = './results/vinet_agg_rec/davis_512/'
mask_root = './DAVIS_demo/Annotations/480p/'
fg_root = './DAVIS_demo/JPEGImages/480p/'
save_root = './results/vinet_agg_rec/'

for mode in ['horizontal', 'vertical']:
    for vid_name in os.listdir(bg_root):

        bg_dir = os.path.join(bg_root, vid_name)
        mask_dir = os.path.join(mask_root, vid_name)
        fg_dir = os.path.join(fg_root, vid_name)

        bg_frames = os.listdir(bg_dir)
        bg_frames.sort()

        for img_name in bg_frames:
            iid = img_name.split('.')[0]
            bg = cv2.imread(os.path.join(bg_dir, iid+'.png'))
            mask = np.array(Image.open(os.path.join(mask_dir, iid+'.png')).convert('P'),np.uint8)
            mask = cv2.resize(mask, (H, W), cv2.INTER_NEAREST)

            [hs, ws] = np.where(mask>0)
            try:
                h1, h2, w1, w2 = min(hs), max(hs), min(ws), max(ws)
                if mode == 'horizontal':
                    bg_half = cv2.resize(bg, (int(H*ratio),W))
                elif mode == 'vertical':
                    bg_half = cv2.resize(bg, (H,int(W*ratio)))

                fg = cv2.resize(cv2.imread(os.path.join(fg_dir, iid+'.jpg')),(H,W), cv2.INTER_CUBIC)
                fg[:,:,0] = fg[:,:,0]*(mask>0)
                fg[:,:,1] = fg[:,:,1]*(mask>0)
                fg[:,:,2] = fg[:,:,2]*(mask>0)

                ori = [(h1+h2)/2,(w1+w2)/2]
                if mode == 'horizontal':
                    x1 = ori[1]*ratio
                    x2 = (W-ori[1])*ratio + ori[1]
                    fg_cut = fg[:,int(x1):int(x2),:]
                elif mode == 'vertical':
                    x1 = ori[0]*ratio
                    x2 = (H-ori[0])*ratio+ori[0]
                    fg_cut = fg[int(x1):int(x2),:,:]
                assert(fg_cut.shape == bg_half.shape)

                fg_cut[fg_cut==0] = bg_half[fg_cut==0]
                save_dir = os.path.join(save_root, mode+'_'+str(ratio), vid_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, iid+'.png'), fg_cut)
            
            except:
                pass

print('Retargeting restuls saved at %s.'%(save_root))




