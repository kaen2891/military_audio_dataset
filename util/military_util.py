from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T

from .augmentation import augment_raw_audio

__all__ = ['get_annotations', 'save_image', 'get_torchaudio', 'generate_fbank', 'get_score']


def get_annotations(args, annotation_file):
    labels = pd.read_csv(annotation_file)
    
    outputs = []
    
    file_path = labels['path'].values.tolist()
    file_label = labels['label'].values.tolist()
    
    return file_path, file_label

def save_image(image, fpath):
    save_dir = os.path.join(fpath, 'image.jpg')
    cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# ==========================================================================


# ==========================================================================
""" data preprocessing """

def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data


def get_torchaudio(args, data_path):
    data, sr = torchaudio.load(data_path)
    if sr != args.sample_rate:
        resample = T.Resample(sr, args.sample_rate)
        data = resample(data)
    
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
        
    padded_sample_data = cut_pad_sample_torchaudio(data, args)

    return padded_sample_data


def generate_fbank(args, audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    if args.model in ['ast']:
        mean, std = -4.2677393, 4.5689974
    else:
        mean, std = fbank.mean(), fbank.std()
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy() #T, F, 1
    #print('fbank', fbank.shape)
    return fbank 


# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    acc = hits[0] / (counts[0] + 1e-10) * 100

    if pflag:
        # print("************* Metrics ******************")
        print("Acc: {}".format(acc))

    return acc
# ==========================================================================
