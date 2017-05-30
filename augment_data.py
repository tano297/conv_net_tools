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

  Example: images=[img],n_rot=4,ccw_limit=90,cw_limit=90
  Returns: [img1: 90 degrees rot ccw,
            img2: 45 degrees rot ccw,
            img3: original,
            img4: 45 degrees rot cw,
            img5: 90 degrees rot cw]            ]
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

def apply_horiz_shear(images,n_shear,max_shear,crop_center=True):
  """
  Applies a horizontal shear transform to every image in the list "images" 
  n_shear times, with the max shear passed in the max_shear argument. By 
  default, we crop the center of the images so that all images have the same 
  shape, but this can be changed with the arg crop_center set to False.

  Returns a list with all the shear samples. Size will be n_shear+1, because
  we also want the original sample to be included. 

  Example: images=[img],n_shear=2, max_shear=0.5
  Returns: [img1=img,
           img2=img with shear of 0.25 in x,
           img3=img with shear of 0.5 in x]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the shear steps
  step_rel = max_shear / float(n_shear) #relative to image size

  # container for sheared images
  sheared_images = []

  #get every image and apply the number of desired shears
  for img in images:
    #get rows and cols to shear
    rows,cols,depth = img.shape
    step_abs = step_rel * cols #absolute to image size

    #shear the amount of times we want
    for i in xrange(0, n_shear+1):
      #create shear matrix by mapping points
      size_inc = abs(int(step_abs*i)) #increases in size in each dim of the img

      #shear is different if is positive or negative
      if int(step_abs*i) > 0:
        pts1 = np.float32([[0,0],[0,rows],[cols,rows]])
        pts2 = np.float32([[0,0],[size_inc,rows],[cols+size_inc,rows]])
        M = cv2.getAffineTransform(pts1,pts2)
        shear_img = cv2.warpAffine(img,M,(cols+size_inc,rows),flags=cv2.INTER_CUBIC)
      elif int(step_abs*i) < 0:
        pts1 = np.float32([[0,rows],[0,0],[cols,rows]])
        pts2 = np.float32([[0,rows],[size_inc,0],[cols,rows]])
        M = cv2.getAffineTransform(pts1,pts2)
        shear_img = cv2.warpAffine(img,M,(cols+size_inc,rows),flags=cv2.INTER_CUBIC)
      else:
        shear_img = img
      #shear using the matrix (and bicubic interpolation)
      
      if crop_center:
        row_start = (shear_img.shape[0]/2) - (rows/2)
        col_start = (shear_img.shape[1]/2) - (cols/2)
        shear_img = shear_img[row_start:row_start+rows,col_start:col_start+cols]

      #append to sheared images container
      sheared_images.append(shear_img)

  return sheared_images

def apply_vert_shear(images,n_shear,max_shear,crop_center=True):
  """
  Applies a vertical shear transform to every image in the list "images" 
  n_shear times, with the max shear passed in the max_shear argument. By 
  default, we crop the center of the images so that all images have the same 
  shape, but this can be changed with the arg crop_center set to False.

  Returns a list with all the shear samples. Size will be n_shear+1, because
  we also want the original sample to be included. 

  Example: images=[img],n_shear=2, max_shear=0.5
  Returns: [img1=img,
           img2=img with shear of 0.25 in y,
           img3=img with shear of 0.5 in y]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the shear steps
  step_rel = max_shear / float(n_shear) #relative to image size

  # container for sheared images
  sheared_images = []

  #get every image and apply the number of desired shears
  for img in images:
    #get rows and cols to shear
    rows,cols,depth = img.shape
    step_abs = step_rel * rows #absolute to image size

    #shear the amount of times we want
    for i in xrange(0, n_shear+1):
      #create shear matrix by mapping points
      size_inc = abs(int(step_abs*i)) #increases in size in each dim of the img

      #shear is different if is positive or negative
      if int(step_abs*i) > 0:
        pts1 = np.float32([[0,0],[cols,0],[cols,rows]])
        pts2 = np.float32([[0,0],[cols,size_inc],[cols,rows+size_inc]])
        M = cv2.getAffineTransform(pts1,pts2)
        shear_img = cv2.warpAffine(img,M,(cols,rows+size_inc),flags=cv2.INTER_CUBIC)
      elif int(step_abs*i) < 0:
        pts1 = np.float32([[0,0],[cols,0],[0,rows]])
        pts2 = np.float32([[0,size_inc],[cols,0],[0,rows+size_inc]])
        M = cv2.getAffineTransform(pts1,pts2)
        shear_img = cv2.warpAffine(img,M,(cols,rows+size_inc),flags=cv2.INTER_CUBIC)
      else:
        shear_img = img
      #shear using the matrix (and bicubic interpolation)
      
      if crop_center:
        row_start = (shear_img.shape[0]/2) - (rows/2)
        col_start = (shear_img.shape[1]/2) - (cols/2)
        shear_img = shear_img[row_start:row_start+rows,col_start:col_start+cols]

      #append to sheared images container
      sheared_images.append(shear_img)

  return sheared_images

def apply_horiz_flip(images):
  """
  Applies a horizontal flip to every image in the list "images" 
  Returns a list with all the original and flipped samples.
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  flipped_images = []

  #get every image and apply the number of desired shears
  for img in images:
    #append original and flipped images to container
    flipped_images.append(img)
    flipped_images.append(cv2.flip(img,1))
 
  return flipped_images

def apply_vert_flip(images):
  """
  Applies a vertical flip to every image in the list "images" 
  Returns a list with all the original and flipped samples. 
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  flipped_images = []

  #get every image and apply the number of desired shears
  for img in images:
    #append original and flipped images to container
    flipped_images.append(img)
    flipped_images.append(cv2.flip(img,0))

  return flipped_images

def apply_gaussian_noise(images,mean,std):
  """
  Applies gaussian noise to every image in the list "images" with the desired

  Returns a list with all the original and noisy images. 
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  noisy_images = []

  #get every image and apply the number of desired shears
  for img in images:
    #get rows and cols apply noise to
    rows,cols,depth = img.shape
    
    # append original image 
    noisy_images.append(img)

    #fill in the per-channel mean and std
    m = np.full((1, depth),mean)
    s = np.full((1, depth),std)

    # add noise to image
    # noisy_img = img.copy()
    noisy_img = np.zeros((rows,cols,depth),dtype=np.uint8)
    noisy_img = cv2.randn(noisy_img,m,s)
    shim.im_plt(noisy_img)
    noisy_img = img + noisy_img

    #append noisy image to container
    noisy_images.append(noisy_img)

  return noisy_images

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
    'times, with ccw_limit as counterclockwise limit, and cw_limit as '+
    'clockwise limit. Angles limited to 180 degrees, and n_rots to 36000'
  )
  parser.add_argument(
    '--horiz_shear',
    nargs='*',
    type=float,
    help='List that contains [n_shear,max_shear]. '+
    'Shears data n_shear times horizontally, with max_shear as limit. '+
    'The limit is defined as a portion of the original image to move -> {0;1}'
  )
  parser.add_argument(
    '--vert_shear',
    nargs='*',
    type=float,
    help='List that contains [n_shear,max_shear]. '+
    'Shears data n_shear times vertically, with max_shear as limit. '+
    'The limit is defined as a portion of the original image to move -> {0;1}'
  )
  parser.add_argument(
    '--gaussian_noise',
    nargs='*',
    type=float,
    help='List that contains [mean,std] for the Gaussian noise applied'
  )
  parser.add_argument(
    '--vert_flip',
    dest='vert_flip',
    action='store_true',
    help='Apply vertical flip. Defaults to False'
  )
  parser.set_defaults(vert_flip=False)
  parser.add_argument(
    '--horiz_flip',
    dest='horiz_flip',
    action='store_true',
    help='Apply horizontal flip. Defaults to False'
  )
  parser.set_defaults(horiz_flip=False)

  #show data on screen?
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

  # horizontal and vertical shearing sanity check. Limiting to 100 shears, 
  # and 10 times the image size in maximum shear. More than this is probably a 
  # mistake (image looks like mom's spaghetti)
  
  #horizontal
  if FLAGS.horiz_shear and (len(FLAGS.horiz_shear) != 2):
    print("Wrong usage of horizontal shear parameters. Check again. Exiting")
    quit()

  n_horiz_shear = 0
  if FLAGS.horiz_shear:
    n_horiz_shear = int(FLAGS.horiz_shear[0])
    max_horiz_shear = FLAGS.horiz_shear[1]
    if n_horiz_shear > 100:
      print("Too many horizontal shears. Exiting")
      quit()
    if abs(max_horiz_shear) > 10:
      print("Horizontal shear size off boundaries. Exiting")
      quit()

  if FLAGS.vert_shear and (len(FLAGS.vert_shear) != 2):
    print("Wrong usage of vertical shear parameters. Check again. Exiting")
    quit()

  n_vert_shear = 0
  if FLAGS.vert_shear:
    n_vert_shear = int(FLAGS.vert_shear[0])
    max_vert_shear = FLAGS.vert_shear[1]
    if n_vert_shear > 100:
      print("Too many vertical shears. Exiting")
      quit()
    if abs(max_vert_shear) > 10:
      print("Vertical shear size off boundaries. Exiting")
      quit()

  # Gaussian noise sanity check
  if FLAGS.gaussian_noise and (len(FLAGS.gaussian_noise) != 2):
    print("Wrong usage of gaussian noise parameters. Check again. Exiting")
    quit()

  # parameter show.   
  print("----------------------------Parameters-------------------------------")
  print("in_dir: ",FLAGS.in_dir)
  print("in_im: ",FLAGS.in_dir)
  print("out_dir: ",FLAGS.out_dir)
  print("rots: ",FLAGS.rots)
  print("horiz_shear: ",FLAGS.horiz_shear)
  print("vert_shear: ",FLAGS.vert_shear)
  print("gaussian_noise",FLAGS.gaussian_noise)
  print("vert_flip",FLAGS.vert_flip)
  print("horiz_flip",FLAGS.horiz_flip)
  print("show: ",FLAGS.show)
  print("---------------------------------------------------------------------")

  # get all cv2 images from dir or input image
  if FLAGS.in_dir:
    files = [ f for f in listdir(FLAGS.in_dir) if isfile(join(FLAGS.in_dir,f))]
    images = [cv2.imread(join(FLAGS.in_dir,img), cv2.IMREAD_UNCHANGED) for img in files]
  else:
    images = [cv2.imread(join(FLAGS.in_img), cv2.IMREAD_UNCHANGED)]

  # apply pertinent transformations
  transformed_list = []

  #rots
  if n_rots:
    print("Rotating images %d times, with ccw_limit:%.2f, and cw_limit:%.2f"
        % (n_rots, ccw_limit, cw_limit))
    rot_list = apply_rotations(images,n_rots,ccw_limit,cw_limit)
    transformed_list.extend(rot_list)
    print("Done!")
  if n_horiz_shear:
    print("Shearing images horizontally %d times, with max_shear:%.2f" 
        % (n_horiz_shear, max_horiz_shear))
    horiz_shear_list = apply_horiz_shear(images,n_horiz_shear,max_horiz_shear)
    transformed_list.extend(horiz_shear_list)
    print("Done!")
  if n_vert_shear:
    print("Shearing images vertically %d times, with max_shear:%.2f" 
        % (n_vert_shear, max_vert_shear))
    vert_shear_list = apply_vert_shear(images,n_vert_shear,max_vert_shear)
    transformed_list.extend(vert_shear_list)
    print("Done!")
  if FLAGS.horiz_flip:
    print("Flipping images horizontally")
    horiz_flip_list = apply_horiz_flip(images)
    transformed_list.extend(horiz_flip_list)
    print("Done!")
  if FLAGS.vert_flip:
    print("Flipping images vertically")
    vert_flip_list = apply_vert_flip(images)
    transformed_list.extend(vert_flip_list)
    print("Done!")
  if FLAGS.gaussian_noise:
    print("Applying gaussian noise with mean:%.2f and std:%.2f"
          % (FLAGS.gaussian_noise[0],FLAGS.gaussian_noise[1]))
    noisy_images = apply_gaussian_noise(images,FLAGS.gaussian_noise[0],
                                        FLAGS.gaussian_noise[1])
    transformed_list.extend(noisy_images)
    print("Done!")
  #if FLAGS.lalalala... Other transformations

  # @TODO: Save to disk in out_dir!!!!!!!!!!!!!!!

  # if asked, show the results
  if FLAGS.show:
    for img in transformed_list:
      shim.im_plt(img)
    shim.im_block()


