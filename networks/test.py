import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
#from dataset import MyDataset
import argparse
import math
import cv2
from timeit import default_timer as timer
from vgg_rgb import Net


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


def get_ortho_transform(phase='val'):
    if phase =='train':
        ortho_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.38938358, 0.40221168, 0.33868686],
                                 std=[0.18162281, 0.17278138, 0.16014436])
        ])
    else:
        ortho_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4226, 0.4488, 0.3917],
                             std=[0.1941, 0.1770, 0.1647])
        ])

    return ortho_transform


parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run[default: 200]')
parser.add_argument('--batch_size', type=int, default=4, help='batch size during training[default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate[default: 1e-3]')
parser.add_argument('--save_path', default='.', help='Trained model save path')
parser.add_argument('--data_path', default='/home/ashj/FB_TEST/', help='Dataset folder, default: data/pyrenees')
parser.add_argument('--region', default='bassiero')
FLAGS = parser.parse_args()

no_of_gpus = torch.cuda.device_count()
print('no_of_gpus', no_of_gpus)
print('Testing for {} region'.format(FLAGS.region))
os.system('mkdir -p {}'.format(FLAGS.save_path))
batch_size= FLAGS.batch_size

device = torch.device("cuda:0")
   
model = Net()
model.load_state_dict(torch.load(FLAGS.save_path + '/model.pth'), strict=False)
model.to(device)

TILE_SIZE = 256
ORTHO_SIZE = 512
MAX_N = 1
INPUT_DIR = os.path.join(FLAGS.data_path, FLAGS.region)
print(INPUT_DIR)

# load low res DEM
#hLR = np.loadtxt(INPUT_DIR + '_15m.dem', delimiter = ',', dtype=np.np.float32)
hLR = np.load(INPUT_DIR + '_15m.npy').astype(np.float32)
hshape = hLR.shape
ntiles = (math.ceil(hshape[0]/TILE_SIZE), math.ceil(hshape[1]/TILE_SIZE))
extshape = (ntiles[0]*TILE_SIZE, ntiles[1]*TILE_SIZE)
hLR = cv2.copyMakeBorder(hLR, 0, extshape[0]-hshape[0], 0, extshape[1]-hshape[1], cv2.BORDER_REPLICATE)

# load ortho
ortho = Image.open(INPUT_DIR + '_ortho1m.jpg')
ortho = np.array(ortho)
print(ortho.shape)
ortho = ortho[:,:,::-1]
# ortho -= (89.08896, 92.25447, 75.68488)
ortho = cv2.copyMakeBorder(ortho, 0, 2*(extshape[0]-hshape[0]), 0, 2*(extshape[1]-hshape[1]), cv2.BORDER_REPLICATE) 
# ortho = ortho.transpose((2,0,1))


# shape for INPUT_DIR (data blob is N x C x H x W), set data
Ntotal = ntiles[0]*ntiles[1]
N = min(Ntotal, MAX_N)
hLR_means = np.zeros(N)

# space for output collection
hout = np.zeros(extshape)

# iterate over batches
numBatches = math.ceil(Ntotal/N)
print('Num batches ', numBatches)

transform = transforms.ToTensor()
ortho_transform =  get_ortho_transform()
model.eval()
for bi in range(numBatches):
    
    Nbatch = min(Ntotal - bi*N, N)

    # prepare INPUT_DIR
    for ni in range(Nbatch):    
        print('ni ', ni)
        nt = bi*N + ni
        ti = nt//ntiles[1]
        tj = nt%ntiles[1]
        print(ti, tj)
        opart = ortho[ti*ORTHO_SIZE:(ti+1)*ORTHO_SIZE, tj*ORTHO_SIZE:(tj+1)*ORTHO_SIZE, :]
        save_rgb = opart

        cv2.imwrite('Ortho_{}_{}.jpg'.format(ti,tj), save_rgb)
        
        hpart = hLR[ti*TILE_SIZE:(ti+1)*TILE_SIZE, tj*TILE_SIZE:(tj+1)*TILE_SIZE]
        hmean = np.mean(hpart)

        inData = hpart - hmean
        inData = inData[np.newaxis, ...] # create C dimension (C = 1)

        # print('inData shape', inData.shape)
        # print('opart shape', opart.shape)
        opart = Image.fromarray(opart)
        opart = ortho_transform(opart)
        inData = torch.from_numpy(inData)
        inData = inData.view([1,1,256,256])
        opart = opart.view([1,3,512,512])
        opart = opart.cuda()
        # print('mean', inData.mean())
        # print('inData shape', inData.shape)

        inData = inData.cuda()
        # print(opart)
        # print(inData)
    # run net
        tini = timer()
        out = model(opart)
        out = out.detach().cpu().float().numpy()
        out = np.sum(out, 1)
        dim = out.shape[-1]
        out = out.reshape([dim, dim])
        print('out shape', out.shape)
        tend = timer()
        np.save('out_{}_{}'.format(ti, tj), out)
    # print('Ellapsed time: %g s' % (tend - tini))

    # collect result
    
    # for ni in range(Nbatch): 
    #     nt = bi*N + ni
    #     ti = nt//ntiles[1]
    #     tj = nt%ntiles[1]
