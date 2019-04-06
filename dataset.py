'''
Video generator from multiple sample videos
'''

import glob
import numpy as np 
from math import ceil
from augmentation import speed,shift_channel,shift_hue,bw,blur,artifical_flash
from moviepy.editor import VideoFileClip

#list of all functions for augmentation of videos
process=[speed,shift_channel,shift_hue,bw,blur,artifical_flash]



def video_generator(sample_vid_set,samples=10,split=.5,prob_process=.5):
	sample_count=0;sample=0
	process_len=len(process)
	sample_len=len(sample_vid_set)
	while (sample_count<samples):
		sample_count+=1
		sample_rand=int(np.random.rand()*sample_len)-1
		#accounting for same spawn
		while sample == sample_rand:
			sample_rand=int(np.random.rand()*sample_len)-1
			print("same spawn ({},{}) changed to ({},{})".format(sample,sample,sample,sample_rand))
		sample=sample_rand

		sample_vid=sample_vid_set[sample]
		clip=VideoFileClip(sample_vid,audio=True)

		if np.random.rand()>=split:
			array_process=np.random.random(process_len)
			for _id,i in enumerate(array_process):
				if i>prob_process:
					try:
						clip=process[_id](clip)
					except:
						print("didn't happened..."); continue
		yield(sample_count,clip)



		
'''
# testing code
sample_vid_set=glob.glob('out-clips/*.mp4')
for i,aug_clip in video_generator(sample_vid_set,1000):
	aug_clip.write_videofile('aug_out/%s.mp4'%i)
'''



