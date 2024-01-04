# UOD_Universal_Oneshot_Detection [arXiv](https://arxiv.org/abs/2306.07615)
Official PyTorch implementation for MICCAI 2023 paper:

[UOD: universal one-shot detection of anatomical landmarks](https://github.com/heqin-zhu/UOD_universal_oneshot_detection)  
 [Heqin Zhu](https://scholar.google.com/citations?user=YkfSFekAAAAJ), [Quan Quan](https://scholar.google.com/citations?user=mlTXS0YAAAAJ), [Qingsong Yao](https://scholar.google.com/citations?user=CMiRzlAAAAAJ), [Zaiyi Liu](https://scholar.google.com/citations?user=OkrZX0AAAAAJ), [S. Kevin Zhou](https://scholar.google.com/citations?user=8eNm2GMAAAAJ)  

![results](https://github.com/heqin-zhu/UOD_universal_oneshot_detection/blob/master/images/network.png)

## Introduction
 One-shot medical landmark detection gains much attention and achieves great success for its label-efficient training process. However, existing one-shot learning methods are highly specialized in a single domain and suffer domain preference heavily in the situation of multi-domain unlabeled data. Moreover, one-shot learning is not robust that it faces performance drop when annotating a sub-optimal image. To tackle these issues, we resort to developing a domain-adaptive one-shot landmark detection framework for handling multi-domain medical images, named __Universal One-shot Detection (UOD)__. UOD consists of two stages and two corresponding universal models which are designed as combinations of domain-specific modules and domain-shared modules. In the first stage, a domain-adaptive convolution model is self-supervised learned to generate pseudo landmark labels. In the second stage, we design a domain-adaptive transformer to eliminate domain preference and build the global context for multi-domain data. Even though only one annotated sample from each domain is available for training, the domain-shared modules help UOD aggregate all one-shot samples to detect more robust and accurate landmarks. We investigated both qualitatively and quantitatively the proposed UOD on three widely-used public X-ray datasets in different anatomical domains (i.e., head, hand, chest) and obtained state-of-the-art performances in each domain.


## Train and Test
### Stage I 
```python
CUDA_VISIBLE_DEVICES=1 nohup python3 train_ssl.py --run_dir .runs/stg1  --run_name RUNNAME --config config_ssl.yaml  --oneshot_id_list 3188 126 JPCLN035  --data_list hand head jsrt --model uvgg --batch_size 6 --phase train -x 1 -e 1500 &>log_stg1 &
```
### Stage II
After Stage I, update `pseudo_path` in `config.yaml` with corresponding path.
```python
nohup python3 main.py -p train -d .runs/stg2 -r RUNNAME_stg2 --model DATR -b 4 -e 300 -C config.yaml --sigma 10 --data_list head hand jsrt -g 1 -x 1 --use_layerscale &> log_stg2 &
```

### Summary
```bash
python3 summary.py -v --SDR 2 2.5 3 4  -r .runs/stg2
```

## Citation
```
@inproceedings{zhu2023uod,
  title={UOD: Universal One-Shot Detection of Anatomical Landmarks},
  author={Zhu, Heqin and Quan, Quan and Yao, Qingsong and Liu, Zaiyi and Zhou, S Kevin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={24--34},
  year={2023},
  organization={Springer}
}
```
## LICENSE
[Apache-2.0](LICENSE)

## Acknowledgements
- [GU2Net](https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection)
- [CC2D](https://github.com/MIRACLE-Center/Oneshot_landmark_detection)
