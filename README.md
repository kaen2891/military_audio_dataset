# MAD: A Military Audio Dataset for Situational Awareness and Surveillance
[Journal](https://www.nature.com/articles/s41597-024-03511-w) | [Kaggle](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset) | [BibTeX](#bibtex)


Official code implementation of "MAD: A Military Audio Dataset for Situational Awareness and Surveillance"

## Requirements
Install the necessary packages with: 
```
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
$ pip install -r requirements.txt
```

## Notation (important)
**You can directly download the overall dataset from** [kaggle](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset).

The `mad_dataset_annotation.csv` must be located in `./`
The `training.csv` and `test.csv` file must be located in `./data/MAD_dataset/`


## Data Preparation 
**Please download the dataset from** [kaggle](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset). **We can ignore this part. Go Training part**

###
To download all the audio samples from youtube url.

```
python3 youtube_audio_download.py
```
It takes around 2-3 hours to download all the videos

###
We then extract waveforms using audio segmentation labels.
```
python get_sample.py
```
All the samples must be located in `./data/MAD_dataset/training` and `./data/MAD_dataset/test`


## Training 
To simply train the model, run the shell files in `scripts/`.    
1. **`scripts/military_resnet18_ce.sh`**: Cross-Entropy loss with ResNet18 (w/ pretrained weights on ImageNet) model.
2. **`scripts/military_resnet18_ce_scratch.sh`**: Cross-Entropy loss with ResNet18 (w/o pretrained weights on ImageNet --> from scratch training) model.
3. **`scripts/military_resnet50_ce.sh`**: Cross-Entropy loss with ResNet50 (w/ pretrained weights on ImageNet) model.
4. **`scripts/military_cnn6_ce.sh`**: Cross-Entropy loss with CNN6 (w/ pretrained weights on AudioSet) model.
5. **`scripts/military_efficient_b0_ce.sh`**: Cross-Entropy loss with EfficientNet-B0 (w/ pretrained weights on AudioSet) model.
6. **`scripts/military_ast_ce.sh`**: Cross-Entropy loss with AST model (w/ pretrained weigths on ImageNet & AudioSet).
7. **`scripts/military_patchmix_ce.sh`**: Cross-Entropy loss with AST model (w/ pretrained weigths on ImageNet & AudioSet), where the label depends on the interpolation ratio.
Except for these scripts, there are many scripts in `scripts/`. Please check.

Important arguments for models.
- `--model`: network architecture, see [models](models/)
- `--from_sl_official`: load ImageNet or AudioSet pretrained checkpoint
- `--audioset_pretrained`: load AudioSet pretrained checkpoint and only support AST

Important arugment for evaluation.
- `--eval`: switch mode to evaluation without any training
- `--pretrained`: load pretrained checkpoint and require `pretrained_ckpt` argument.
- `--pretrained_ckpt`: path for the pretrained checkpoint

The pretrained model checkpoints will be saved at `save/[EXP_NAME]/best.pth`.     

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{kim2024military,
  title={A Military Audio Dataset for Situational Awareness and Surveillance},
  author={Kim, June-Woo and Yoon, Chihyeon and Jung, Ho-Young},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={668},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Contact
- June-Woo Kim: kaen2891@gmail.com
