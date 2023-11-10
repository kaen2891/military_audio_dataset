from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy

from .military_util import get_annotations, generate_fbank, get_torchaudio
from .augmentation import augment_raw_audio


class MilitarySoundDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        self.data_folder = args.data_folder
        self.train_flag = train_flag
        self.annotation = os.path.join(args.data_folder, 'training.csv') if self.train_flag else os.path.join(args.data_folder, 'test.csv') 
        
        self.transform = transform
        self.args = args

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels
        
        self.file_path, self.file_label = get_annotations(self.args, self.annotation)

        if print_flag:
            print('*' * 20)  
            print("Extracting military sounds..")
        
        self.class_nums = np.zeros(args.n_cls)
        for sample in self.file_label:
            self.class_nums[sample] += 1
        
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        if print_flag:
            print('[Military dataset information]')
            print('total number of audio data: {}'.format(len(self.file_path)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
        
        
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        for index in range(len(self.file_path)): #for the training set, 4142
            audio, label = self.file_path[index], self.file_label[index]
            
            data = get_torchaudio(args, os.path.join(self.data_folder, audio))
            fbank = generate_fbank(args, data, self.sample_rate, n_mels=self.n_mels)
            self.audio_images.append((fbank, label))
        
        # ==========================================================================

    def __getitem__(self, index):
        audio_image, label = self.audio_images[index][0], self.audio_images[index][1]
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        return audio_image, label

    def __len__(self):
        return len(self.file_path)