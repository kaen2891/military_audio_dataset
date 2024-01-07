# -*- coding: cp949 -*-
from glob import glob
import pandas as pd
import os
import librosa
import soundfile as sf
from torchaudio import transforms as T

SAVED_LIST = []

def cut_and_save(audio, start_t, end_t, file_num, v_num, data_path):
    start_frame = int(int(start_t) * 16000)
    end_frame = int(int(end_t) * 16000)
    
    new_wav = audio[start_frame:end_frame]
    if len(new_wav) > 160000:
        new_wav = new_wav[:160000]
    
    if not os.path.isfile(os.path.join(data_path, str(file_num)+'.wav')):
        sf.write(os.path.join(data_path, str(file_num)+'.wav'), new_wav, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        print('{} is saved'.format(os.path.join(data_path, str(file_num)+'.wav')))
    else:
        print('{} is passed'.format(os.path.join(data_path, str(file_num)+'.wav')))


def extract_wav(saved_path, video_num, file_num, start_t, end_t, DATA_DIR, new_DATA_DIR, ):
    prev_video_num = None
    for i in range(len(video_num)):
        if len(str(video_num[i])) == 1:
            v_num = '00'+str(video_num[i])
        elif len(str(video_num[i])) == 2:
            v_num = '0'+str(video_num[i])
        else:
            v_num = str(video_num[i])
        
        print('v_num', v_num)
        
            
        if v_num != prev_video_num:
        
            new_save_dir = os.path.join(new_DATA_DIR, v_num)        
            if not os.path.exists(new_save_dir):
                os.makedirs(new_save_dir, exist_ok=True)
            
            y, sr = sf.read(os.path.join(DATA_DIR, v_num + '.wav'))
            print('y {} sr {}'.format(y.shape, sr))
            if sr != 16000:
                print('sr is not 16000, : ', sr)
                y, sr = librosa.load(os.path.join(DATA_DIR, v_num + '.wav'), sr=sr)
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                print('after resample, y {}'.format(y.shape))
            else:
                y, sr = librosa.load(os.path.join(DATA_DIR, v_num + '.wav'), sr=sr)
            
            duration = y.shape[0]
            print('duration', duration)
            cut_and_save(y, start_t[i], end_t[i], file_num[i], v_num, new_save_dir)
        
        else:
            cut_and_save(y, start_t[i], end_t[i], file_num[i], v_num, new_save_dir)
        
        prev_video_num = v_num


## For training set
data_dir = './data/MAD_dataset/wav_files/'
train_save_dir = './data/MAD_dataset/training'
train_label_file = './training.csv' #
train_label = pd.read_csv(train_label_file)

saved_path = train_label['path'].values.tolist()
video_num = train_label['video_num'].values.tolist()
start_t = train_label['start_time'].values.tolist()
end_t = train_label['end_time'].values.tolist()
file_num = train_label['file_id'].values.tolist() 

extract_wav(saved_path, video_num, file_num, start_t, end_t, data_dir, train_save_dir) # training set

## For test set
test_save_dir = './data/MAD_dataset/test'
test_label_file = './test.csv' #
test_label = pd.read_csv(test_label_file)

saved_path = test_label['path'].values.tolist()
video_num = test_label['video_num'].values.tolist()
start_t = test_label['start_time'].values.tolist()
end_t = test_label['end_time'].values.tolist()
file_num = test_label['file_id'].values.tolist() 

extract_wav(saved_path, video_num, file_num, start_t, end_t, data_dir, test_save_dir) # test set



    

