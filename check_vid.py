'''
For cross-checking the generated dataset and csv file for scene cut detection
Can be used for manually validating the results of the detection as well  

'''
import pandas as pd
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-no','--video_no',type=str,default='1',
                        help='Video no')
    args = parser.parse_args()


    num=args.video_no

    vid_name='./aug_final.mp4'
    scene_cut=pd.read_csv('csv_aug_data.csv',index_col=0)
    frame_nos=scene_cut['frame_no']
    print(frame_nos)

    start_frames=frame_nos.as_matrix()


    cap = cv2.VideoCapture(vid_name)


    _,frame_prev=cap.read()
    _,frame_curr=cap.read()
    
    h,w,_=frame_curr.shape

    print (cap)
    count=1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame_prev2=frame_prev
            frame_prev=frame_curr
            frame_curr=frame
            panel=np.hstack([frame_prev2,frame_prev,frame_curr])
            panel_resized=cv2.resize(panel, (int(w*1.5),int(h/2.0))) 
            cv2.imshow('Panel',panel_resized)
     
            if count in start_frames:
                print("Frame Change\n")
                cv2.waitKey(1000)

            count+=1
     
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()