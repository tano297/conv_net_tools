#!/usr/bin/python2.7

"""
This code uses opencv to apply different augmentations to a single image or a
training dataset, which is a common practice in machine learning. Functions are
pretty self-explanatory.

General comment for all functions:
  We don't do anything with labels in this functions, they take images as 
  inputs, and return images as output, but in the same order, so label 
  management can be done easily in a higher level.

  e.g: images: [...,img1,img2,...]
       labels: [...,lbl1,lbl2,...]
       index:  [...,i,   i+1 ,...]
       
       n_transformations: 2
       
       returns: [...,img1_1,img1_2,img1_3,img2_1,img2_2,img2_3,...]
       labels:  [...,lbl1,  lbl1,  lbl1,  lbl2,  lbl2,  lbl2,  ...]
       index:   [...,i,     i+1,    ...   i+3,   i+4     ...      ]
                                           ^---> (i+n_transformations+1)

  Indexes for labels should therefore travel in increments of
  (n_transformations+1) in the higher level to recover the labeled dataset 
  structure. Randomize before feeding this to the training, even if the dataset
  has been shuffled before, since all the transformed images will be highly
  correlated with each other, and they will be of the same class!
"""

import argparse
import os
from os import listdir
from os.path import isfile, join
import random
from shutil import rmtree as sh_rmtree
from shutil import copy as sh_copy
import math

#opencv is needed for the image transformations
import cv2

#matplotlib to plot samples from main
import matplotlib.pyplot as plt


def apply_rotations(images,n_rot,cw_limit,ccw_limit):
  """
  Rotates every image in the list "images" n_rot times, between cw_limit 
  (clockwise limit) and ccw_limit (counterclockwise limit). The limits are 
  there to make sense of the data augmentation. E.g: Rotating an mnist digit
  180 degrees turns a 6 into a 9, which makes no sense at all.
  cw_limit and ccw_limit are in degrees!

  Returns a list with all the rotated samples. Size will be n_rot+1, because
  we also want the original sample to be included
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the initial angle and the step
  initial_angle = float(ccw_limit)
  step_angle = (float(cw_limit) + float(ccw_limit)) / float(n_rot)

  # container for rotated images
  rotated_images = []

  #get every image and apply the number of desired rotations
  for img in images:
    #get rows and cols to rotate
    rows,cols,depth = img.shape
    #rotate the amount of times we want them rotated
    for i in xrange(0, n_rot+1):
      #create rotation matrix with center in the center of the image,
      #scale 1, and the desired angle (we travel clockwise, from ccwlimit)
      M = cv2.getRotationMatrix2D((cols/2,rows/2),initial_angle-i*step_angle,1)
      #rotate using the matrix (using bicubic interpolation)
      rot_img = cv2.warpAffine(img,M,(cols,rows),flags=cv2.INTER_CUBIC)
      #append to rotated images container
      rotated_images.append(rot_img)

  return rotated_images