import argparse, time, os
import imageio
import numpy as np
import math
import options.options as option
from utils import util
from solvers import create_solver as create_solver 
from data import create_dataloader
from data import create_dataset
import math
import cv2
from dem2ply import dem2ply
from timeit import default_timer as timer
import torchvision.transforms as transforms
import torch
from PIL import Image

save_patches = True
#save_patches = False
debug = True
#debug = False
#save_out = True
save_out = False

makeTensor = transforms.ToTensor()

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    pick_factor = np.max(img1) - np.min(img1)
    print(pick_factor)
    mse = np.mean((img1 - img2)**2)
    rmse = math.sqrt(mse)
    if mse == 0:
        return float('inf')

    return 20 * math.log10((pick_factor) / rmse), rmse

parser = argparse.ArgumentParser(description='Test Super Resolution Models')
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt)
opt = option.dict_to_nonedict(opt)
TILE_SIZE = 200
ORTHO_SIZE = 2 * TILE_SIZE
MAX_N = 1
# initial configure
scale = opt['scale']
degrad = opt['degradation']
network_opt = opt['networks']
model_name = network_opt['which_model'].upper()
if opt['self_ensemble']: model_name += 'plus'

# create solver (and load model)
solver = create_solver(opt)
# Test phase
print('===> Start Test')
print("==================================================")
print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

#region = 'durrenstein'
#region = 'montemagro'
#region = 'forcanada'
#region = 'bassiero'
use_uniform = False
#use_uniform = True
logs = []
error_list= {}
error_list['durrenstein'] = [63.6, 0.901]
error_list['forcanada'] = [62.0,1.097]
error_list['montemagro'] = [71.1,0.587]
error_list['bassiero'] = [63.4, 1.005]
regions = ['durrenstein', 'montemagro', 'forcanada', 'bassiero']
for region in regions:
    if debug:
       region = 'forcanada'
       #region = 'montemagro'
    
    INPUT_DIR = '/home/ashj/FB_TEST/'+region

    ortho = Image.open(INPUT_DIR + '_ortho1m.jpg')
    
    if use_uniform:
       region = 'uniform'
       INPUT_DIR = '/home/ashj/FB_TEST/'+region
    hLR = np.load(INPUT_DIR + '_15m.npy')
    ortho = makeTensor(ortho)
    ortho = np.array(ortho, dtype=np.float32).transpose([1,2,0])

    #ortho = makeTensor(ortho)

    hshape = hLR.shape
    ntiles = (math.ceil(hshape[0]/TILE_SIZE), math.ceil(hshape[1]/TILE_SIZE))
    extshape = (ntiles[0]*TILE_SIZE, ntiles[1]*TILE_SIZE)
    hLR = cv2.copyMakeBorder(hLR, 0, extshape[0]-hshape[0], 0, extshape[1]-hshape[1], cv2.BORDER_REPLICATE)
    hHR = np.load(INPUT_DIR + '_2m.npy')
    hHR = cv2.copyMakeBorder(hHR, 0, extshape[0]-hshape[0], 0, extshape[1]-hshape[1], cv2.BORDER_REPLICATE)

    ortho = ortho[:,:,::-1]
    ortho = cv2.copyMakeBorder(ortho, 0, 2*(extshape[0]-hshape[0]), 0, 2*(extshape[1]-hshape[1]), cv2.BORDER_REPLICATE) 
    #ortho = ortho.transpose((2,0,1))
    #ortho = ortho[:,:,::-1]

#    print('ortho shape', ortho.shape)
    # shape for INPUT_DIR (data blob is N x C x H x W), set data
    Ntotal = ntiles[0]*ntiles[1]
    N = min(Ntotal, MAX_N)
    hLR_means = np.zeros(N)

    # space for output collection
    hout = np.zeros(extshape)
    need_HR = False
    # iterate over batches
    numBatches = math.ceil(Ntotal/N)
 #   print('Num batches ', numBatches)
    transform = transforms.ToTensor()
    data = {}
    for bi in range(numBatches):
        
        Nbatch = min(Ntotal - bi*N, N)

        # prepare INPUT_DIR
        for ni in range(Nbatch):    
            # print('ni ', ni)
            nt = bi*N + ni
            ti = nt//ntiles[1]
            tj = nt%ntiles[1]
#            print(ti, tj)
    #        ti, tj = 6, 12
    #        ti, tj = 3, 3
            hpart = hLR[ti*TILE_SIZE:(ti+1)*TILE_SIZE, tj*TILE_SIZE:(tj+1)*TILE_SIZE]
            opart = ortho[ti*ORTHO_SIZE:(ti+1)*ORTHO_SIZE, tj*ORTHO_SIZE:(tj+1)*ORTHO_SIZE,:]
            save_rgb = opart * 255
            GT = hHR[ti*TILE_SIZE:(ti+1)*TILE_SIZE, tj*TILE_SIZE:(tj+1)*TILE_SIZE]

            if save_patches:   
                 cv2.imwrite('Ortho_{}_{}.jpg'.format(ti,tj), save_rgb)
                 np.save('LR_{}_{}'.format(ti, tj), hpart)
                 np.save('GT_{}_{}'.format(ti, tj), GT)

            hmean = np.mean(hpart)

            inData = hpart - hmean

            inData = inData[np.newaxis, ...] # create C dimension (C = 1)

            inData = torch.from_numpy(inData)
            inData = inData.view([1,1,TILE_SIZE,TILE_SIZE])

            inData = inData.cuda()
            opart = makeTensor(opart)

            #opart = opart[np.newaxis, ...]
            opart = opart.view([1, 3, ORTHO_SIZE, ORTHO_SIZE])
            opart = opart.cuda()


            data['LR'] = inData
            data['LR_path'] = 'temp'
            data['ortho'] = opart
            solver.feed_data(data, need_HR=False)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            # total_time.append((t1 - t0))

            visuals = solver.get_current_visual2(need_HR=False)
            # sr_list.append(visuals['SR'])

            tend = timer()
            
            # hpart = out[0] # remove also C dimension 
            hpart = visuals['SR'][3] 
            hpart = hpart + hmean
            hout[ti*TILE_SIZE:(ti+1)*TILE_SIZE, tj*TILE_SIZE:(tj+1)*TILE_SIZE] = hpart[...]
            if save_patches:      
                for idx in range(4):
                    out_tile = visuals['SR'][idx] + hmean
                    # print('max at', np.max(np.array(out_tile)))
                    # print(out_tile)
                    np.save('vis_{}_{}_{}'.format(ti, tj, idx), out_tile[0][0])
            if debug:
               exit()
    # crop out the added padding
    hout = hout[0:hshape[0], 0:hshape[1]]
    hLR = hLR[0:hshape[0], 0:hshape[1]]
    hHR = hHR[0:hshape[0], 0:hshape[1]]
   # print('hout shape', hout.shape)

    print('Region: ', region)
    psnr, rmse = calculate_psnr(hHR, hout)
#    print('For output psnr: {}, rmse: {}'.format(psnr, rmse))
    logs.append([region, "PSNR: " , psnr, error_list[region][0], "RMSE:", rmse, error_list[region][1]])
    print(logs[-1]) 
    psnr, rmse = calculate_psnr(hHR, hLR)
   # print('For input psnr: {}, rmse: {}'.format(psnr, rmse))
    #logs.append(["Bicubic", psnr, rmse])
    # save output
    if save_out:
         np.save(INPUT_DIR + '_ami', hout)
    #dem2ply(hout, INPUT_DIR + '_out', 2)


for l in logs:
    print(l)


np.save('test_logs', logs)
