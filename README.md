## Crop_Diseases
Crop Diseases Detection

The code is derived from the Google recognition API, with some modifications based on the data.

Deep learning framework Tensorflow1.9

[Dataset Download] (https://pan.baidu.com/s/1ey3ioopiJZu1-SV-neH2Ng) Password: yq30

[Download the pre-trained model] (https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

### Generate TFrecords

Run process.py to compress the data image to generate a TFRecords type data file, which can improve the data reading efficiency

`` `
#Modify the main path of process.py to its own compressed path after downloading

python process.py
`` `
### Training model
`` `

# Configure train.sh parameters
#Generated TFrecords Road King (modify according to your actual situation, the same below)
DATASET_DIR = / media / en / DATA / AgriculturalDisease20181023 / tf_data
#Models generated during the training process, iteratively saved data locations
TRAIN_DIR = / media / en / DATA / AgriculturalDisease20181023 / check_save / resnetv1_101_finetune
#Define the pre-trained model definition (pre-trained model download address is given above)
CHECKPOINT_PATH = / media / en / DATA / AgriculturalDisease / check / resnet_v1_101.ckpt

python train_image_classifier1.py \
    --train_dir = $ {TRAIN_DIR} \
    --dataset_name = AgriculturalDisease \
    --dataset_split_name = train \
    --dataset_dir = $ {DATASET_DIR} \
    --model_name = resnet_v1_101 \ #Model name
    --checkpoint_path = $ {CHECKPOINT_PATH} \ # Pre-trained model position, this is not needed if fully initialized training
    --learning_rate = 0.001 \ #Learning rate
    --checkpoint_exclude_scopes = resnet_v1_101 / logits \ #Use pre-training to exclude the last classification layer (because it is not the same as your data classification)
    --max_number_of_steps = 40000 \ #Number of iterations


After the file is configured, execute the script
sh train.sh
`` `


### Test model

`` `
# Configure test.sh parameters
# Test data location
DATASET_DIR = / media / en / DATA / AgriculturalDisease20181023 / tf_data
# Model generated after training
CHECKPOINT_PATH = / media / en / DATA / AgriculturalDisease20181023 / check_save / vgg16_finetune / model.ckpt-40000


python test.py \
    --alsologtostderr \
    --checkpoint_path = $ {CHECKPOINT_FILE} \
    --dataset_dir = $ {DATASET_DIR} \
    --dataset_name = AgriculturalDisease \
    --dataset_split_name = validation \
    --model_name = vgg_16 \
    --checkpoint_path = $ {CHECKPOINT_PATH}
`` `

### Feature map visualization
`` `
# Also modify your own model path, the picture path is in the demo.py file, you can modify it yourself to get the features of the vgg_16 network to extract the middle layer, so you need to train a vgg16 network
# If you want to use other networks, you need to define a network similar to vgg_16_layer. The original network does not return the information of the middle layer.
sh demo.sh
`` `

### Curve generation

Using the tensorboard tool to read the path of the training generated file, you can get the data change of the training process.
`` `
# You can download the data of the chart and draw it by yourself using matlab or python
tensorboard --logdir = 'xxxx'
`` `