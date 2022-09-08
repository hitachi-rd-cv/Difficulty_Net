# Difficulty-Net: Learning to Predict Difficulties for Long-Tailed Recognition
____

This contains the code for our WACV 2023 paper available at [arXiv]. 

### Requirements
___
The environment required to successfully reproduce our results mainly needs
```
- Python >= 3.6
- PyTorch == 1.6.0
- yacs == 0.1.8
```
### How to run the code
___

### CIFAR-LT

Our implementation on CIFAR100-LT dataset is built on the code of [CDB-CE] [2]. 

To run code for CIFAR-LT, go to folder CIFAR-LT.

Download the dataset from [here] and save it in the ```data/``` folder. 

To start stage-1 training, 

```
python cifar_train_exaugment.py --imbalance 200 --loss_type CE --n_gpus 1 --lamda 0.3
```

This starts the stage-1 training with Difficulty-Net based weighted CE loss. The imbalance variable can take values in {10, 20, 50, 100, 200}.

To use Balanced Softmax with our method, use ```--loss_type Balanced_Softmax``` . The best model will be saved in ```saved_model/``` .

To start stage-2 training,

```
python cifar_train_exaugment_stage2.py --imbalance 200 --stage_2_method LAS --stage1_trained_model ... --n_gpus 1
```

For evaluation,

```
python cifar_test.py --saved_model_path ... --imbalance 100 --n_gpus 1
```

### ImageNet-LT/Places-LT

This code is built on that of [MiSLAS] [1]. 

Change directory to ```ImageNet_Places-LT/``` .

After downloading [ImageNet] or [Places], update the data path in the respective ```config/{dataset}/***.yaml``` files

To start our stage-1 training, 

```
python train_stage1.py --cfg config/{dataset}/{dataset}_{model}_stage1.yaml

```
For stage-2 training,
```
python train_stage2.py --cfg config/{dataset}/{dataset}_{model}_stage2.yaml resume saved/{dataset}_{model}_stage1_{yyyymmddhhmm}/ckps/model_best.pth.tar
```
Change the stage-1 path to resume from as needed.

To evaluate the best model, use
```
python eval.py --cfg config/{dataset}/{dataset}_{model}_stage2.yaml resume saved/{dataset}_{model}_stage2_{yyyymmddhhmm}/ckps/model_best.pth.tar
```
Put the respective dataset and model name in ```{dataset}``` and ```{model}``` respectively.  Currently, this code is only supported for ```resnet10 ``` and ```resnet50``` on ```imagenet``` and ```resnet152``` on ```places```. 

### References:
1. Zhong et. al., Improving Calibration for Long-Tailed Recognition, CVPR 2021
2. Sinha et. al., Class-Wise Difficulty-Balanced Loss for Solving Class-Imbalance, ACCV 2020




[MiSLAS]: https://github.com/dvlab-research/MiSLAS
[ImageNet]: https://image-net.org/index.php
[PLaces]: http://places2.csail.mit.edu/index.html
[CDB-CE]: https://github.com/hitachi-rd-cv/CDB-loss
[here]: https://www.cs.toronto.edu/~kriz/cifar.html
[arXiv]: https://github.com/hitachi-rd-cv/Difficulty_Net





   

   
