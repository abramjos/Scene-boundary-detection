# Scene-boundary-detection in Videos
Implementation of the paper 'Ridiculously Fast Shot Boundary Detection with Fully Convolutional Neural Networks' from scratch.
Paper can be found at https://arxiv.org/abs/1705.08214

3D CNN is extensively used in this CNN model to achieve the stated performance. hence the model is fullt convolutional in time.

# Prerequisites
* Python
* Keras (with tensorflow-gpu preffered)
* Moviepy
* Opencv,Numpy,Pandas

The samples videos should be snippets of the video scenes based on the scene boundary or shot-cut, preferably kept in 'out-clips/'.

# Augmentation class and helper files
  Augmentation class works using the moviepy(used for editing the videos) and it offer an effective library to augment the dataset of videos.
  it contains:
## dataset_generator.py
  Creates the dataset from the multiple augmentations listed in the 'augmentation_helper.py'.\n
  Creates a video and a csv file which includes the scene boundary frame numbers.
## augmentation_helper.py
  A helper file that has several functions to augment the dataset which includes many real to life scenerios including artificial flash mentioned in the paper.
## sample_video_csv_gen.py
  Sample use case of 'dataset_generator.py', creates the files aug_final.mp4 and csv_aug_data.csv. 

# Training the model
The model is an implementation of 10 frames/predcition model from the paper which gives one output for 10 frames.
Video augmented data is not required as long as you can provide a csv(with 'frame_no,cut and transistion' colums) and video file. 
## model.py
The script for training the model, files aug_final.mp4 and csv_aug_data.csv has to be provided. The model uses 'adam' as optimizer and 'categorical crossentropy' for calculating loss.

Tensorboard and model checkpoints are used.
![](https://github.com/abramjos/Scene-boundary-detection/blob/master/model.jpg)
## datagen.py & epoch_generator.py
Both the fles are to handle the image queue for the training purpose.
epoch_generator.py ensures that the data fed into the model is equalized(equal no of postitive and negative dataset).


# Testing
## test_model.py
  The script is to test the model performance using the generated model weights after training, ie the 'cut_video_final.h5'.
  Provides a image stream of 10 images and corresponding prediction.
## check_vid.py
  'check_vid.py' will provide a visualization for a .csv with scene cut frames and a video corresponding to it.
  As test_model.py it also provide a image stream of 10 images and corresponding prediction.


# TO DO
- [X] Augmentation for artificial lighting, blurness, speed, color-channel(hue,BW and channel switch)
- [ ] Augmentation for paning and zooming.
- [ ] A generic model for scalable operation to reduce redundancy(any no of frames/ many prediction).
