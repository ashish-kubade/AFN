import torch.utils.data as data
import os
import numpy as np
from data import common
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        print('Inside data loader')
        print(self.opt)
        self.lr_size = opt['LR_size']
        self.train = (opt['phase'] == 'train')
        path = opt['data_path']
        print('path', path)
        if(self.train):
            self.file_names = np.load(os.path.join(path, 'train_files.npy'))
        else:
            self.file_names = np.load(os.path.join(path, 'val_files.npy'))

        self.hr_dem = os.path.join(path, 'HR')
        self.lr_dem = os.path.join(path, 'LR')
        self.ortho_path = os.path.join(path, 'ortho')
        self.makeTensor = transforms.ToTensor()

    def __getitem__(self, index):
        name = self.file_names[index]
        # print('input name', name)
        ortho_name = name[:-4]+ '.jpg'

        LR_DEM = np.load(os.path.join(self.lr_dem, name))
        HR_DEM = np.load(os.path.join(self.hr_dem, name))
        ortho = Image.open(os.path.join(self.ortho_path, ortho_name))
        seed = random.randrange(0, LR_DEM.shape[0] - self.lr_size + 1)

        LR_DEM = LR_DEM[..., np.newaxis]
        HR_DEM = HR_DEM[..., np.newaxis]
        LR_DEM = self._get_patch(LR_DEM, self.lr_size, 1, seed)
        HR_DEM = self._get_patch(HR_DEM, self.lr_size, 1, seed)
    
        # print('LR_DEM shape', LR_DEM.shape)
        #print('HRshape', HR_DEM.shape)
        # print('Input shapes, ', LR_DEM.shape)
        # print('HR_DEM', HR_DEM.shape)
        LR_DEM, HR_DEM = common.np2Tensor([LR_DEM, HR_DEM], self.opt['rgb_range'])
        ortho = self.makeTensor(ortho)
#        print('ortho 1', ortho[:, :20, :20])
        ortho = ortho.numpy().transpose([1,2,0])
#        print('ortho shape',ortho.shape)
 
#       print('ortho 2', ortho[:, :20, :20])
        ortho = self._get_patch(ortho, self.lr_size, 2, seed)

#        print('ortho 3', ortho[:, :20, :20])
        ortho = self.makeTensor(ortho)
        # LR_DEM = LR_DEM.transpose(2,0,1)
        # HR_DEM = HR_DEM.transpose(2,0,1)
 #       print('ortho 4', ortho[:, :20, :20])
        input_dict = {'LR': LR_DEM,
                      'HR': HR_DEM,
                      'ortho': ortho,
                      'LR_path': name,
                      'HR_path' : name
                     }


        # Give subclasses a chance to modify the final output
        #print('Able to return data')
        return input_dict

    def __len__(self):
        return len(self.file_names)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        hr = common.read_img(hr_path, self.opt['data_type'])

        return lr, hr, lr_path, hr_path


    def _get_patch(self, img, tar_shape, scale, seed):
       
        # random crop and augment
        patch = common.get_patch(img, tar_shape, scale, seed)

        # print('shapes again here', lr.shape, hr.shape)
#        lr, hr = common.augment([lr, hr])
        # lr = common.add_noise(lr, self.opt['noise'])
        # print('shapes again', lr.shape, hr.shape)
        return patch
