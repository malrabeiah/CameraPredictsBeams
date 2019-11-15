# mmWave Base Stations with Cameras:
Deep learning solutions are developed to tackle mmWave beam and link blockage predictions using camera feed. For more information, please refer to (mmWave Base Stations with Cameras: Vision Aided Beam and Blockage Prediction)[https://arxiv.org/abs/1911.06255].

# Dependencies:
1) Python 3.7 

2) Pytorch 1.3

3) NVIDIA GPU with a compatible CUDA toolkit (see [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)).

4) Processed ViWi dataset (see [ViWi wesite](https://viwi-dataset.net/))

# Running the code:
The scripts available here are manily for training and testing a modified ResNet-18 model for mmWave beam prediction. To train and test, you need to do the following:

1) Prepare two sets of data using the ViWi framework. One set is for training and the other is for testing (For more information on the data structure, see the next section).

2) Set the paths to the training and testing sets in the script "main.py" (i.e., modify train_dir and val_dir to point to you sets).

3) Set the path to where you want the trained network to be saved, by modifying net_name.

5) Run main.py

The script finishes training and testing, it will save the accuracies in a result.mat file, and it will store the trained network.

# Data Structure:
The script assumes a training and testing sets of data structured as a directory of subdirectories, as follows:
```
training_data
  |
  |- 1
  |- 2
  |- 3
  .
  .
  .
  |- x
 ```
where x = the size of the beam-forming codebook. The name of each sub-directory refers to the beam index in the codebook, and the contents of each sub-directory "j", for examples, are the images of those users served with the jth beam in the codebook.


