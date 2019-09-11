# FSANet:Ultra-Deep Neural Network for Face Anti-Spoofing

This repository contains an implementation of "Ultra-Deep Neural Network for Face Anti-Spoofing". 
For a detailed description of the architecture please read [our paper](https://link.springer.com/chapter/10.1007/978-3-319-70096-0_70).
Please cite the paper if you use the code from this repository in your work.

### Bibtex

```
@inproceedings{tu2017ultra,
  title={Ultra-deep neural network for face anti-spoofing},
  author={Tu, Xiaokang and Fang, Yuchun},
  booktitle={International Conference on Neural Information Processing},
  pages={686--695},
  year={2017},
  organization={Springer}
}
```

##Running the Code

###Prerequisites
```
tensorflow >=1.14.0
Keras >= 2.2.4 
opencv >= 3.4.2 
h5py >= 2.8.0
pandas >= 0.25.1
matplotlib >= 3.1.1
tqdm >= 4.32.1
```

###Train from scratch

```
python preprocess.py
python train.py --nb_epoch 100 --input_num 25 --dataset casia
```

###Test

```
python eval.py --input_num 25 --dataset casia --checkpoint_path /path/to/checkpoint 
```
