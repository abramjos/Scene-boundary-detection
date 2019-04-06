'''
Code for generating the augmented dataset resulting a .mp4 file and .csv file. 
'''

import glob
import numpy as np 
from math import ceil
from augmentation import speed,shift_channel,shift_hue,bw,blur,artifical_flash
from moviepy.editor import VideoFileClip,concatenate_videoclips
import pandas as pd
from math import ceil
from dataset_generator import video_generator

#gathering the video samples to be augmented and generated
sample_vid_set=glob.glob('out-clips/*.mp4')


cols=['frame_no','cut','transition']
csv_data=pd.DataFrame(columns=cols)

#using the generator
for i,aug_clip in video_generator(sample_vid_set,samples=100):
	print(i,aug_clip.duration)
	aug_frames=int(aug_clip.fps*aug_clip.duration)

	if i==1:
		final_clip = aug_clip
	else:
		final_frames=int(final_clip.fps*final_clip.duration)
		csv_thread=pd.Series(data=[final_frames,1,0],index=cols)
		csv_data=csv_data.append(csv_thread,ignore_index=True)
		final_clip = concatenate_videoclips([final_clip,aug_clip])
		print(ceil(final_clip.duration))


final_clip.write_videofile("aug_final.mp4")
csv_data.to_csv('csv_aug_data.csv')




# concat_clip = mp.concatenate_videoclips(clips, method="compose")
# concat_clip.fps=24
# concat_clip.write_videofile("out/x6.mp4")

