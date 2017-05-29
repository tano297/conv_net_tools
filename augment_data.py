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
import show_img as shim
import numpy as np

#opencv is needed for the image transformations
import cv2

#matplotlib to plot samples from main
import matplotlib.pyplot as plt


def apply_rotations(images,n_rot,ccw_limit,cw_limit):
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Apply augmentations to data. See source")
  parser.add_argument(
    '--in_dir',
    type=str,
    help='Path of raw dataset to augment. No default'
  )
  parser.add_argument(
    '--in_img',
    type=str,
    help='Path of raw image to augment. No default'
  )
  parser.add_argument(
    '--out_dir',
    type=str,
    default='/tmp/out_dir',
    help='Path of augmented output dataset. Defaults to \'%(default)s\''
  )
  parser.add_argument(
    '--rots',
    nargs='*',
    type=float,
    help='List that contains [n_rots,ccw_limit,cw_limit]. Rotates data n_rots '+
    'times, with ccw_limit as counterclockwise limit, and cw_limit as '
    'clockwise limit. Angles limited to 180 degrees, and n_rots to 36000'
    'Defaults to \'%(default)s\''
  )
  parser.add_argument(
    '--show',
    dest='show',
    action='store_true',
    help='Show all augmented data on screen. Defaults to False'
  )
  parser.set_defaults(show=False)

  #parse args
  FLAGS, unparsed = parser.parse_known_args()

  # Sanity checks
  # Input directory needs OR input image needs to be provided. Both shouldn't
  # work either. Done with dirty implementation of an xnor gate
  if not (bool(FLAGS.in_dir) != bool(FLAGS.in_img)):
    print("Dataset directory OR single image path needs to be provided. "+
          "Exiting")
    quit()

  # It also needs to exist
  if FLAGS.in_dir and (not os.path.exists(FLAGS.in_dir)):
    print("Input directory needs to exist, Einstein :) Exiting")
    quit()

  if FLAGS.in_img and (not os.path.exists(FLAGS.in_img)):
    print("Input image needs to exist, Einstein :) Exiting")
    quit()

  # rotations sanity check: limit angles to 180 and n_rots to 36000)
  # this is chosen somewhat randomly, I cannot imagine somebody willingly
  # expanding an image with more resolution than 0.01 degrees, so it is probably
  # a coding error and it will stupidly all the memory.
  if FLAGS.rots and (len(FLAGS.rots) != 3):
    print("Wrong usage of rotation parameters. Check again. Exiting")
    quit()

  n_rots = 0
  if FLAGS.rots:
    n_rots = int(FLAGS.rots[0])
    ccw_limit = FLAGS.rots[1]
    cw_limit = FLAGS.rots[2]
    if n_rots > 36000:
      print("Too many rotations. Exiting")
      quit()
    if cw_limit > 180 or ccw_limit > 180:
      print("Rotations off boundaries. Exiting")
      quit()

  # parameter show.   
  print("----------------------------Parameters-------------------------------")
  print("in_dir: ",FLAGS.in_dir)
  print("in_im: ",FLAGS.in_dir)
  print("out_dir: ",FLAGS.out_dir)
  print("rots: ",FLAGS.rots)
  print("show: ",FLAGS.show)
  print("---------------------------------------------------------------------")

  # get all cv2 images from dir or input image
  if FLAGS.in_dir:
    files = [ f for f in listdir(FLAGS.in_dir) if isfile(join(FLAGS.in_dir,f))]
    images = [cv2.imread(join(FLAGS.in_dir,img), cv2.IMREAD_UNCHANGED) for img in files]
  else:
    images = [cv2.imread(join(FLAGS.in_img), cv2.IMREAD_UNCHANGED)]

  print("len images: ", len(images))
  print images

  # apply pertinent transformations
  transformed_list = []

  #rots
  if n_rots:
    print("Rotating images %d times, with ccw_limit:%.2f, and cw_limit:%.2f"
        % (n_rots, ccw_limit, cw_limit))
    rot_list = apply_rotations(images,n_rots,ccw_limit,cw_limit)
    transformed_list.extend(rot_list)
    print("rot list type",type(rot_list))
    print("rot list length: ",len(rot_list))
    print("tf list length: ",len(transformed_list))
    print("Done!")
  #if FLAGS.lalalala... Other transformations

  # @TODO: Save to disk in out_dir!!!!!!!!!!!!!!!

  # if asked, show the results
  if FLAGS.show:
    for img in transformed_list:
      shim.im_plt(img)
    shim.im_block()


