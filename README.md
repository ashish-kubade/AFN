# Introduction

This repository contains the source code to replicate the results in our paper **[AFN: Attentional Feedback Network based 3D Terrain Super-Resolution](https://arxiv.org/abs/2010.01626?context=cs)** at [ACCV 2020, Kyoto.](http://accv2020.kyoto/)

# System/Software Requirements
1. PyTorch: 1.4
2. Nvidia-1080Ti with 11GB of VRAM


# Inference
1. To get an inference with our pretrained model on your custom model, please add the test data pairs (aerial image and low-resolution DEM) in the datasets directory.
2. Download the pretrained model from [this path](https://drive.google.com/drive/folders/15zxbCVT-9UMYeUD3oKXj8Tp0aeMeXUum?usp=sharing).
3. Update the paths accordingly in options/test/test_options.json file. 
4. Use following command to test.
  ```
python test.py -opt options/test/test_options.py
  ```

# Train
1. To train the network on your own dataset, please add the training pairs into datasets directory. 
2. Update the paths accordingly in options/train/train_options.json file. You can play with the hyperparameters like *number of steps: T*, *number of groups: N* in the same file.
3. For training, use follwing command:

```
python train.py -opt options/train/train_options.py
```


<h1>
Citation
</h1>
If you find this work useful in your research, please consider citing with:

```
@inproceedings{kubade2020afn,
  title={AFN: Attentional Feedback Network based 3D Terrain Super-Resolution},
  author={Kubade, Ashish and Patel, Diptiben and Sharma, Avinash and Rajan, KS},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```
