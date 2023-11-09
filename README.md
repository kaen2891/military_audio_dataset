# MAA: Dataset Related to Military Activity Audio for Acoustic Hazardous Situation Surveillance System
Official code implementation of "MAA: Dataset Related to Military Activity Audio for Acoustic Hazardous Situation Surveillance System"

## Requirements
Install the necessary packages with: 
```
$ pip install torch torchvision torchaudio
$ pip install -r requirements.txt
```

## Data Preparation
Download the MAA dataset files from [figshare](https://figshare.com/account/home).     
And extract zip file into ./data/ folder
All the waveform `*.wav` files and label `*.csv` files should be saved in `data/MSC_dataset/`

Note that the MSC dataset consists of a total of dataset consists of a total of 7,46 audio clips, 
of which 6,429 contain training samples and 1,037 contain test samples.


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
- `--model`: network architecture, see [models](models/)s
- `--from_sl_official`: load ImageNet or AudioSet pretrained checkpoint
- `--audioset_pretrained`: load AudioSet pretrained checkpoint and only support AST

Important arugment for evaluation.
- `--eval`: switch mode to evaluation without any training
- `--pretrained`: load pretrained checkpoint and require `pretrained_ckpt` argument.
- `--pretrained_ckpt`: path for the pretrained checkpoint

The pretrained model checkpoints will be saved at `save/[EXP_NAME]/best.pth`.     

## Result


## Contact
- June-Woo Kim: kaen2891@knu.ac.kr