# Building & Training an Image-to-Image Generative Model

The code on this repo is an adaptation of the Pytorch pix2pix implementation ([repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)) and the Tensorflow implementation of HED detector ([repo](https://github.com/harsimrat-eyeem/holy-edge))

The models implemented are based on:
* Isola, et al. 2016. Image-to-Image Translation with Conditional Adversarial Networks. [arxiv](https://arxiv.org/abs/1611.07004)
* Xie, et al. 2015. Holistically-Nested HED Detection. [arxiv](https://arxiv.org/abs/1504.06375)

## Quick Commands
### Resize a folder of images
First edit the input and output folders in [/dataset_creation/reduce_folder.sh](/dataset_creation/reduce_folder.sh).

`sh reduce_folder.sh`

### Extract Canny Edges from a folder
```
cd dataset_creation
python2 auto_canny.py -i [input folder] -o [output folder]
```

### Extract HED Edges from a folder
First edit the input and output folders in [/holy-edge/hed/config/hed.yaml]. 
Create a list of the files from which to extract the edges outside the folder: 

```
cd [image_folder]
cd ..
touch test.lst
ls [image_folder] > test.lst
```

And then run the HED detector
```
cd holy-edge
CUDA_VISIBLE_DEVICES=0 python2 run-hed --test --config-file ./hed/config/hed.yaml
```

### Generate images with a pre-trained Image-to-Image model
Edit the necessary parameters in [/predict.py](/predict.py)
```
python2 predict.py --exp_name [experiment_name]
```

### Train an Image-to-Image Model
```
python2 train.py --exp_name [experiment_name] --dataroot_faces [source images folder] --dataroot_edges [edges image folder]
```
Training results and model checkpoints will be saved in `runs/experiment_name`

## Experiments
### CelebA - centered and cropped
#### No adversarial loss
![Alt text](/img/lambda_0.png?raw=true "Optional Title")

#### Canny edges - Lambda 1e-4
##### Train Set
![Alt text](/img/canny-centered-lambda_1e-4.png?raw=true "Optional Title")
##### Test Set
![Alt text](/img/test-canny-centered-lambda_1e-4.png?raw=true "Optional Title")

#### Canny edges - Lambda 1e-3
##### Train Set
![Alt text](/img/canny-centered-lambda_1e-3.png?raw=true "Optional Title")
##### Test Set
![Alt text](/img/test_canny-centered-lambda_1e-3.png?raw=true "Optional Title")

#### HED edges - Lambda 5e-4
##### Train Set
![Alt text](/img/hed-centered-lambda_5e-4.png?raw=true "Optional Title")
##### Test Set
![Alt text](/img/test-hed-centered-lambda_5e-4.png?raw=true "Optional Title")

#### HED edges - Lambda 1e-1
##### Train Set
![Alt text](/img/hed-centered-lambda_1e-1.png?raw=true "Optional Title")
##### Test Set
![Alt text](/img/test-hed-centered-lambda_1e-1.png?raw=true "Optional Title")

#### HED edges - Lambda 1e-2
##### Test Set
![Alt text](/img/hed-centered-lambda0-01_patch_size.png?raw=true "Optional Title")

### CelebA - faces in the wild
#### HED edges - Lambda 1e-3
##### Train Set
![Alt text](/img/hed-wild-lambda0-001_patch_size30.png?raw=true "Optional Title")
##### Test Set
![Alt text](/img/test-hed-wild-lambda0-001_patch_size30.png?raw=true "Optional Title")
