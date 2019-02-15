## Workflow for training/adapting the network on cloud

### use the following snippets to load the model in google colab and train the network

to verify once

0 following files must be present in the colab

resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
pretrained_gmn.h5
cells.zip

1 clone the repo
!git clone https://github.com/segwitt/class-agnostic-counting.git

2 mount your drive into the colab by authentication

from google.colab import drive
drive.mount('/content/drive')

3 copy the cells.zip file to current directory
!cp ./drive/My\ Drive/cells.zip ./

4 extract the files

!mkdir cells
import zipfile
zip_ref = zipfile.ZipFile('./cells.zip', 'r')
zip_ref.extractall('./cells')
zip_ref.close()

5 a bit of cleaning

!mv ./class-agnostic-counting/* ./
!rm -rf class-agnostic-counting
!ls

6 move the cells folder to the src directory

!mv cells ./src
!ls

7 move the pretrained_gmn.h5 weight to the ./src/models direc

!mkdir ./src/models
!cp ./drive/My\ Drive/pretrained_gmn.h5 ./src/models
!ls ./src

8 move the resnets weights (doubtful whether necessary or not)

!cp ./drive/My\ Drive/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 ./src/models
!ls ./src/models

9 adapt the pretrained weights
!python src/main.py --mode adapt --batch_size 4 --epoch 4 --dataset vgg_cell --data_path src/cells --gmn_path src/models/pretrained_gmn.h5

