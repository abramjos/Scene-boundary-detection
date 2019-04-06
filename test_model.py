'''
Sample code for prediction and displaying the scene cut using the trained weight file. 
'''

import cv2
import argparse
import numpy as np
from math import ceil,floor
import pandas as pd 
from model import model


class datagen():
    def __init__(self,no_frames,model,model_weight,video_file=None,csv_file=None):
        if video_file==None:
            input_video=input('Enter Video name')
        self.extra_frames=ceil(no_frames/2.0)
        self.csv_file=csv_file
        self.cap=cv2.VideoCapture(video_file)
        self.len=int(self.cap.get(7))
        channel=3
        self.panel_pipe=np.zeros((no_frames,128,128,channel))
        self.image_pipe=np.zeros((no_frames,64,64,channel))
        self.prediction=0,0
        self.model_3d=model
        self.model_3d.load_weights(model_weight)

    def _image_insert(self,frame_64,frame_128):
        self.image_pipe=np.append(self.image_pipe[1:],[frame_64],axis=0)
        self.panel_pipe=np.append(self.panel_pipe[1:],[frame_128],axis=0)
        return()

    def _create_pannel(self):
        panel=np.hstack(self.panel_pipe)
        h,w,_=panel.shape
        panel_image=cv2.line(panel,(w/2,0),(w/2,h),(255,255,255),4)
        panel_text=np.ones((20,w,3))        
        panel_text=cv2.rectangle(panel_text,(1,1),(w,20),(255,255,255),thickness=cv2.FILLED)
        panel_text=cv2.putText(panel_text, 'Prediction val:{}'.format(self.prediction[0],self.prediction[1]), (w/2-10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), lineType=cv2.LINE_AA)
        panel_final=np.vstack([panel_image,panel_text]) 
        return(panel_final)

    def test_set(self):

        _,init_image=self.cap.read()
        frame_64=cv2.resize(init_image,(64,64),cv2.INTER_LINEAR).astype(np.float32)
        frame_64/=255.
        frame_128=cv2.resize(init_image,(128,128),cv2.INTER_LINEAR)

        self.image_pipe=np.tile(frame_64,(10,1,1,1))
        self.panel_pipe=np.tile(frame_128,(10,1,1,1))

        width = init_image.shape[1]
        height = init_image.shape[0]
        channel = init_image.shape[2]

        count=1  
        while(self.cap.isOpened()):
#           for i in range(self.extra_frames):
            ret, frame = self.cap.read()
            if ret==True:
                count+=1
                frame_64=cv2.resize(frame,(64,64),cv2.INTER_LINEAR).astype(np.float32)
                frame_64/=255.
                frame_128=cv2.resize(frame,(128,128),cv2.INTER_LINEAR)
                self._image_insert(frame_64,frame_128)
                image=self.image_pipe.reshape((1,)+self.image_pipe.shape)
                image_reshaped=image.reshape((64, 64, 10, 3))
                prediction=self.model_3d.predict(image_reshaped)
                predictionX=(prediction[0]>.5).astype(np.uint8)
                self.prediction=predictionX[1],predictionX[0]
                panel=self._create_pannel()
                cv2.imshow('panel',panel)
                if predictionX[0]==1:
                    cv2.imwrite('./swap_pred/{}.jpg'.format(count),panel)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return('Done')

if __name__ == '__main__':
    #loading the model and passing it for prediction and returns CSV results
    train_model=model()
    model=train_model.model_3d
    model_weight='cut_video_final.h5'

    test_gen=datagen(no_frames=10,model=model,model_weight=model_weight,video_file='../dataset_creator/aug_final.mp4')
    test_gen.test_set()