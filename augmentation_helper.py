
'''
Contains all the augmentation functions
'''

import numpy as np
from math import sqrt,exp,floor,ceil
from moviepy.editor import vfx
import cv2
from random import shuffle


def scroll(get_frame, t):
    """
    This function returns a 'region' of the current frame.
    The position of this region depends on the time.
    """
    frame = get_frame(t)
    frame_region = frame[int(t):int(t)+360,:]
    return frame_region

def speed(clip,val_min=2,val_max=5):
    """
    This function returns a video at a random speed
    """
    
    speed=min(.5,max(abs(np.random.normal(loc=1.0)),val_max))
    return(clip.fx(vfx.speedx,speed))

def shift_channel(clip):
    """
    This function returns a video with a randomized colour channels,
     to compensate background changes
    """
    channel=[0,1,2]
    shuffle(channel)
    def pipe(image):
        return image[:,:,channel]
    return(clip.fl_image(pipe))

def shift_hue(clip,h_max=10):
    """
    This function returns a video with random hue changes
    """
    if h_max==None:
        return (x)
    def pipe(x,h_max=10):
        h_shift=int(np.random.normal(-h_max,h_max))
        x=x.astype(np.uint8)
        hsv=cv2.cvtColor(x,cv2.COLOR_BGR2HSV)
        h=hsv[:,:,0]
        h_shift=h+h_shift
        h_rot=h_shift+180
        h=h_rot%180
        hsv[:,:,0]=h
        x_shift=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return(x_shift)
    return(clip.fl_image(pipe))

def bw(clip,chance=.1):
    ''''
    Returns black and white once in 10 times, can change value with randomness.
    '''
    if np.random.rand()<chance:
        return(clip.fx(vfx.blackwhite))
    else:
        return(clip)

def blur(clip):
    ''''
    Returns temporal blured videos that imitates out of focus cases in videos.
    '''
    def sort(array, num_peaks=2, start_ascending=True):
        if num_peaks is None:
            num_peaks = len(array) // 6
        sorted_ar = sorted(array)
        subarrays = [sorted_ar[i::num_peaks] for i in range(num_peaks)]
        for i, subarray in enumerate(subarrays, start=int(not start_ascending)):
            if i % 2:
                # subarrays are in ascending order already!
                subarray.reverse()
        return sum(subarrays, [])

    rand=np.random.rand(1,np.random.randint(int(.2*clip.duration*clip.fps),int(.5*clip.duration*clip.fps)))
    rand=np.sort(rand[0]*10)
    start=int(np.random.uniform(0,0.5*clip.duration)*clip.fps)
    randx=[i+i%2+1 for i in np.array(rand).astype(np.uint8)]
    array=np.ones(int(ceil(clip.fps*clip.duration))).astype(np.uint8)
    array[start:start+len(randx)]=randx

    def pipe(image,frame_no):
        return(cv2.GaussianBlur(image,(array[frame_no],array[frame_no]),0))
    return(clip.fl(lambda gf,t : pipe(gf(t),int(t*clip.fps))))


def artifical_flash(clip):
    ''''
    Returns artificial flash scenerios in the video.
    ##gamma value suitable gamma_max=1.5
    '''
    def gamma(image,frame_no):
        gamma=array[frame_no]
        image=image.astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
        image_gamma=cv2.LUT(image, table).astype(np.uint8)
        return (image_gamma)

    def sort(array, num_peaks, start_ascending=True):
        if num_peaks is None:
            num_peaks = len(array) // 6
        sorted_ar = sorted(array)
        subarrays = [sorted_ar[i::num_peaks] for i in range(num_peaks)]
        for i, subarray in enumerate(subarrays, start=int(not start_ascending)):
            if i % 2:
                # subarrays are in ascending order already!
                subarray.reverse()
        return sum(subarrays, [])

    def flash(image,frame):
        if frame>=start and frame<(start+len_dist):
            image_flash=gamma(image,sample_dist[int(frame-start)])
            return(image_flash)
        else:
            return(image)

    rand_i=np.random.randint(0,clip.fps/2)
    samples=np.random.rand(1,2+int(rand_i*clip.duration))

    start=int(np.random.uniform(0.1*clip.duration,0.6*clip.duration)*clip.fps)

    flash_intensity=int(np.random.uniform(5,7))

    samples=sort(array=samples[0]*flash_intensity,num_peaks=np.random.randint(2,10))

    #randx=[i+i%2+1 for i in np.array(samples).astype(np.uint8)]
    array=np.ones(int(ceil(clip.fps*clip.duration)))
    array[start:start+len(samples)]=samples
    # import ipdb;ipdb.set_trace()

    #sample_dist=gaussian(sampling,7)
    #len_dist=len(sample_dist)

    return(clip.fl(lambda gf,t : gamma(gf(t),int(t*clip.fps))))



