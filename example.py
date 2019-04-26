# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import os
import numpy as np
import time
import cv2
import sys
from noise_model import *

LIMIT_FRAMES = 1000
DELAY = 0.01 # must be > 0

# RegEx for the object IDs of the objects of interest (that is, the objects
# around which you want the bounding boxes).
REGEX_OBJECTS_OF_INTEREST = [
    "Touareg[\w]*",
    "BMW850[\w]*",
    "bighorse",
    "smlhorse",
    "sheep1",
    "sheep2",
    "sheep3",
    "sheep4",
    "sheep5",
    "sheep6",
    "mountainbike",
    "van",
    "giraffe",
    "dalmation",
    "boxer"
]

# Classes corresponding to the objects in the REGEX_OBJECTS_OF_INTEREST list above.
OBJECT_OF_INTEREST_CLASSES = [
    "car",
    "car",
    "horse",
    "horse",
    "sheep",
    "sheep",
    "sheep",
    "sheep",
    "sheep",
    "sheep",
    "mountainbike",
    "truck",
    "giraffe",
    "dog",
    "dog"
]

# An exhaustive list of all classes
CLASSES = ['mountainbike',
           'car',
           'truck',
           'dog',
           'horse',
           'sheep',
           'giraffe']

# Confusion matrix for object misclassification emulation
CONFUSION_MATRIX = np.array(
        [[0.85714286,0.,0.14285714,0.,0.,0.,0.],
        [0.06666667,0.8,0.06666667,0.,0.,0.06666667,0.],
        [0.,0.,1.,0.,0.,0.,0.],
        [0.03636364,0.,0.01818182,0.94545455,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.01408451,0.,0.,0.98591549]]).T
            


##########################################################################
##########################################################################
##########################################################################

MESH_COLS = np.array([
          [57 ,181, 55,255],
          [6  ,108,153,255],
          [191,105,112,255],
          [72 ,121, 89,255],
          [64 ,225,190,255],
          [59 ,190,206,255],
          [36 ,13 , 81,255],
          [195,176,115,255],
          [27 ,171,161,255],
          [180,169,135,255],
          [199,26 ,29 ,255],
          [239,16 ,102,255],
          [146,107,242,255],
          [23 ,198,156,255],
          [160,89 ,49 ,255],
          [116,218,68 ,255],
        ])

# deal with swapped RB values
cp = MESH_COLS.copy()
MESH_COLS[:,0] = cp[:,2]
MESH_COLS[:,2] = cp[:,0]

client = None

        

def getImages(airsimRequestList):
    """Returns images from airsim
    
    Parameters
    ----------
    airsimRequestList : list[airsim.ImageRequest]
        The airsim requests to fetch.
    
    Returns
    -------
    list[np.ndarray]
        List of images.
    """
    ims = [] 

    for response in client.simGetImages(airsimRequestList):
        im = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        im = im.reshape(response.height, response.width, 4)           # reshape array to 4 channel image 
        im = np.flipud(im)                                            # original image is flipped vertically
        ims.append(im)

    return ims


def setup():

    # Clear background
    msg = 'Setting everything to ID 255 (to clear unwanted objects)...'
    print(msg, end='')
    found = client.simSetSegmentationObjectID("[\w]*", 255, True);
    print(' ' * (65 - len(msg)) + ('[SUCCESS]' if found else '[FAILED!]'))

    # Set objects of interest
    for key, val in zip(REGEX_OBJECTS_OF_INTEREST, range(len(REGEX_OBJECTS_OF_INTEREST))):
        msg = 'Setting %s to ID %d...' % (key, val)
        print(msg, end='')
        found = client.simSetSegmentationObjectID(key, val, True);
        print(' ' * (40 - len(msg)) + ('[SUCCESS]' if found else '[FAILED!]'))


def swapRB(im):
    cp = im.copy() 
    cp[:,:,0] = im[:,:,2]
    cp[:,:,2] = im[:,:,0]
    return cp


def mainloop(quiet=False):
    try:
        for i in range(LIMIT_FRAMES):

            sys.stdout.write('\b'*27 + 'Running %d/%d '%(i, LIMIT_FRAMES) + ('[-  ] ','[ - ] ', '[  -] ', '[ - ] ')[i%4])
            sys.stdout.flush()

            # get segmentation image and true image
            im, trueIm = getImages([
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False), 
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            # find unique colors and draw bounding boxes
            colours = np.unique(im.reshape((-1,4)), axis=0)

            # store the BBs and their classes
            bbs = []
            classes = []

            for col in colours:
                colours_of_interest = np.sum(np.all(MESH_COLS == col, axis=-1))
                
                # ignore if this colour does not correspond to an object of interest.
                if colours_of_interest == 0:
                    continue
                elif colours_of_interest > 1:
                    print("[WARNING] Multiple objects have the same color in segmented view! Using lowest index...")

                index = np.where(np.all(MESH_COLS == col, axis=-1))[0][0]
                objClass = OBJECT_OF_INTEREST_CLASSES[index]

                mask = np.all(im == col, axis=-1)
                locs = np.array(np.where(mask))
                
                # find the BB
                min_x = np.min(locs[0,:])
                max_x = np.max(locs[0,:])
                min_y = np.min(locs[1,:])
                max_y = np.max(locs[1,:])
                bbs.append((min_x, max_x, min_y, max_y))
                classes.append(objClass)

            bbs_clean = np.array(bbs).copy()
            

            #################################################
            ##### Add noise to the BBs
            #################################################
            # first do some mis-classification
            classes = misclassify(classes, CONFUSION_MATRIX)
            # now add some error to the BBs
            bbs = add_jitter(bbs, shape=im.shape, length_scale_fraction=0.05, center_error_fraction=0.05)
            bbs, classes = introduce_false_negatives(bbs, classes, p=0.01, min_size=4)
            bbs, classes = introduce_false_positives(bbs, CLASSES, classes, shape=im.shape, p=0.01)
            bbs, classes = merge_close_bbs(bbs, classes, area_similarity_factor=1.7, overlap_factor=0.5)
            #################################################

            if not quiet:
                # Draw the images
                boxedTrue = draw_bbs_on_image(trueIm, bbs)
                boxedSeg =  draw_bbs_on_image(im, bbs)

                # display the images
                cv2.imshow('Scene + BBs', swapRB(np.flipud(boxedTrue)))
                cv2.imshow('Segmented + BBs', swapRB(np.flipud(boxedSeg)))
                cv2.waitKey(int(DELAY * 1000))

            #################################################
            ##### Do something with the BBs for this frame...
            #################################################
            # write the bbox locations to files
            # with open('../results/images/boxed/boxed_%d.dat' % (i,), 'w') as outFile:
            #     for bb_clean, bb, cls in zip(bbs_clean, bbs, classes):
            #         # x and y are swapped
            #         outFile.write('%d, %d, %d, %d, ' % (bb_clean[2], bb_clean[3], bb_clean[0], bb_clean[1]))
            #         outFile.write('%d, %d, %d, %d, ' % (bb[2], bb[3], bb[0], bb[1]))
            #         outFile.write('%s\n' % (cls,))
            #################################################

    except KeyboardInterrupt:
        print('\b'*20 + 'Keyboard interrupt caught, cleaning up and saving data...')
        print('Done! Exiting...')




if __name__ == '__main__':
    import airsim

    client = airsim.VehicleClient()
    client.confirmConnection()
    setup()

    # To increase the speed, redice the size of the output video
    mainloop(quiet=False)
