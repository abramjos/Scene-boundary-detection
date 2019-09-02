import keras 

from keras.models import Sequential
from keras.layers import Conv3D,InputLayer,Dense,Activation,MaxPool3D,Flatten,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.metrics import mean_squared_error,binary_accuracy,categorical_crossentropy,categorical_accuracy
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard

from datagen import datagen
import numpy as np
from tqdm import tqdm
import pandas as pd
from epoch_generator import generator_epoch

class model():

    def __init__(self,no_frames=10,depth=3,out_classes=2):
        self.out_classes=out_classes
        self.no_frames=no_frames
        self.depth=depth
        self.model_3d=self.model()
        print(self.model_3d.summary())
    
    # Model with 10 input frames and 1 prediction. 
    # TO DO
    #   ~ a generic model network to acomodate any no frames

    def model(self):
        input_layer=(64,64,self.no_frames,self.depth)
        kernel_conv1=(5,5,3)
        kernel_conv2=(3,3,3)
        kernel_conv3=(6,6,1)
        kernal_softmax=(1,1,4)

        model = Sequential()
        model.add(InputLayer(input_layer))
        model.add(Conv3D(kernel_size=kernel_conv1, filters=16, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv2, filters=24, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv2, filters=32, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv3, filters=12, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(MaxPool3D((1,1,4)))
        model.add(Flatten())
        model.add(Dense((self.out_classes),activation='softmax'))

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[categorical_crossentropy,categorical_accuracy])

        return(model)

    # Training the model with the video and csv dataset
    def train(self, nb_epoch=1000, batch_size=32
        ,video_file='../dataset_creator/aug_final.mp4',csv_file='../dataset_creator/csv_aug_data.csv'):
        gen=datagen(no_frames=10,video_file=video_file,csv_file=csv_file)
        image_data,cut=[],[]

        try:
            image_data=np.load('im_data.npy')
            cut=np.load('cut.npy')
            print("data has been loaded")

        except:
            for image_64,prediction in tqdm(gen.data_gen(),total=gen.len):
                image_data.append(image_64.reshape((64, 64, 10, 3)))
                cut.append(np.array(prediction))

            image_data=np.array(image_data)
            cut=np.array(cut)
            np.save('im_data.npy',image_data)
            np.save('cut.npy',cut)

        cut=np.argmin(cut,axis=1)
        print("data has been converted")
        image_train,image_test=image_data[:int(.8*len(cut))],image_data[int(.8*len(cut)):]
        cut_train,cut_test=cut[:int(.8*len(cut))],cut[int(.8*len(cut)):]
    
        gen_train=generator_epoch(image_train,cut_train,batch_size=8)
        gen_test=generator_epoch(image_test,cut_test,batch_size=8)



        weights_file = 'scene_cut.{epoch:02d}-{val_loss:.2f}.h5'
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./tf_log', histogram_freq=0, write_graph=True, write_images=False)
        early_stopper = EarlyStopping(min_delta=0.001, patience=100)
        model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                           save_weights_only=True, mode='auto')

        self.model_3d.fit_generator(generator=gen_train,
                    validation_data=gen_test,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=nb_epoch, verbose=2,
                    callbacks=[early_stopper, model_checkpoint, tbCallBack])

        return("training completed")

    # Testing the output. Using a actual dataset to predict on the performance
    #   and record the results into csv. 
    def test(self):

        image_data=np.load('im_data.npy')
        cut=np.load('cut.npy')
        print("data has been loaded")
        col=['set_no','actual','predicted']
        result=pd.DataFrame(columns=col)
        for _id,image_64 in enumerate(image_data):
            pred_val=self.model_3d.predict((1,)+image_64.shape)
            actual_val=cut(_id)
            print("\nActual value\t:{}, Predicted\t:{}",format(actual_val,pred_val[0]))
            result_thread=pd.Series([_id,actual_val,pred_val],index=col)
            result=result.append(result_thread,ignore_index=True)
        return(result)

if __name__ == '__main__':

    train_model=model()
    train_model.train()
    train_model.model_3d.save('scene_cut_3D.h5')
    res=train_model.test()
    res.to_csv('result.csv')
