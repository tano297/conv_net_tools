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

def apply_horiz_stretch(images,n_stretch,max_stretch,crop_center=True):
  """
  Applies a horizontal stretch transform to every image in the list "images" 
  n_stretch times, with the max stretch passed in the max_stretch argument. 

  Stretch > 1 expands the image, and < 1 compresses the image.

  By default, we crop the center of the images so that all images have the same 
  shape, but this can be changed with the arg crop_center set to False. 

  Returns a list with all the stretch samples. Size will be n_stretch+1, because
  we also want the original sample to be included. 

  Example: images=[img],n_stretch=2, max_stretch=1.5
  Returns: [img1=img,
           img2=img with stretch of 1.25 in x,
           img3=img with stretch of 1.5 in x]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the stretch steps
  step_rel = (max_stretch - 1)/float(n_stretch)  #relative to image size

  # container for stretched images
  stretched_images = []

  #get every image and apply the number of desired stretches
  for img in images:
    #get rows and cols to stretch
    rows,cols,depth = img.shape

    stretched_images.append(img)

    #stretch the amount of times we want
    for i in xrange(1, n_stretch+1):
      #create stretch matrix by mapping points
      new_size = int((step_rel*i+1)*cols) #abs increase in size (or decrease if <1) 
      
      #compress or stretch? (neg vs pos)
      pts1 = np.float32([[0,0],[cols,0],[cols,rows]])
      pts2 = np.float32([[0,0],[new_size,0],[new_size,rows]])
      M = cv2.getAffineTransform(pts1,pts2)
      stretch_img = cv2.warpAffine(img,M,(new_size,rows),flags=cv2.INTER_CUBIC)
            
      if crop_center:
        #different if I stretch or compress.
        if(max_stretch>=1):
          #if stretch, then cut out center 
          row_start = (stretch_img.shape[0]/2) - (rows/2)
          col_start = (stretch_img.shape[1]/2) - (cols/2)
          stretch_img = stretch_img[row_start:row_start+rows,col_start:col_start+cols]
        else:
          #if compress image will be smaller than original
          #fill image with zeros and copy the compressed one in the center
          fix_size_stretch_img = np.zeros(img.shape).astype(np.uint8)
          row_start = (rows - stretch_img.shape[0])/2
          col_start = (cols - stretch_img.shape[1])/2
          fix_size_stretch_img[row_start:row_start+stretch_img.shape[0],
                               col_start:col_start+stretch_img.shape[1]] = stretch_img
          stretch_img = fix_size_stretch_img
      #append to stretched images container
      stretched_images.append(stretch_img)

  return stretched_images

def apply_vert_stretch(images,n_stretch,max_stretch,crop_center=True):
  """
  Applies a vertical stretch transform to every image in the list "images" 
  n_stretch times, with the max stretch passed in the max_stretch argument. 

  Stretch > 1 expands the image, and < 1 compresses the image.

  By default, we crop the center of the images so that all images have the same 
  shape, but this can be changed with the arg crop_center set to False. 

  Returns a list with all the stretch samples. Size will be n_stretch+1, because
  we also want the original sample to be included. 

  Example: images=[img],n_stretch=2, max_stretch=1.5
  Returns: [img1=img,
           img2=img with stretch of 1.25 in y,
           img3=img with stretch of 1.5 in y]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the stretch steps
  step_rel = (max_stretch - 1)/float(n_stretch)  #relative to image size

  # container for stretched images
  stretched_images = []

  #get every image and apply the number of desired stretches
  for img in images:
    #get rows and cols to stretch
    rows,cols,depth = img.shape

    stretched_images.append(img)

    #stretch the amount of times we want
    for i in xrange(1, n_stretch+1):
      #create stretch matrix by mapping points
      new_size = int((step_rel*i+1)*rows) #abs increase in size (or decrease if <1) 
      
      #compress or stretch? (neg vs pos)
      pts1 = np.float32([[0,0],[0,rows],[cols,rows]])
      pts2 = np.float32([[0,0],[0,new_size],[cols,new_size]])
      M = cv2.getAffineTransform(pts1,pts2)
      stretch_img = cv2.warpAffine(img,M,(cols,new_size),flags=cv2.INTER_CUBIC)
            
      if crop_center:
        #different if I stretch or compress.
        if(max_stretch>=1):
          #if stretch, then cut out center 
          row_start = (stretch_img.shape[0]/2) - (rows/2)
          col_start = (stretch_img.shape[1]/2) - (cols/2)
          stretch_img = stretch_img[row_start:row_start+rows,col_start:col_start+cols]
        else:
          #if compress image will be smaller than original
          #fill image with zeros and copy the compressed one in the center
          fix_size_stretch_img = np.zeros(img.shape).astype(np.uint8)
          row_start = (rows - stretch_img.shape[0])/2
          col_start = (cols - stretch_img.shape[1])/2
          fix_size_stretch_img[row_start:row_start+stretch_img.shape[0],
                               col_start:col_start+stretch_img.shape[1]] = stretch_img
          stretch_img = fix_size_stretch_img
      #append to stretched images container
      stretched_images.append(stretch_img)

  return stretched_images

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
    noisy_img = img + noisy_img

    #append noisy image to container
    noisy_images.append(noisy_img)

  return noisy_images

def apply_occlusions(images,grid_x,grid_y,selection):
  """
  Applies a grid to each image and removes a block from selection (zeroing it).
  Returns a list with all the original and occluded images. 

  Example: grid_x=3,grid_y=3,selection=[1,3]. 

  This divides the image in 9, and returns the original image with the 1st 
  and 3rd quadrant occluded (in separate images).

  Grid for this case: 
  -------------------------
  |   0   |   1-< |   2   |
  -------------------------
  |   3<- |   4   |   5   |
  -------------------------
  |   6   |   7   |   8   |
  -------------------------

  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  occluded_images = []

  #get every image and apply the number of desired shears
  for img in images:
    # append original image 
    occluded_images.append(img)

    #get rows and cols
    rows,cols,depth = img.shape
    
    #number of rows and cols in subsections
    x_subsec = cols / grid_x
    y_subsec = rows / grid_y

    for idx in selection:
      # select x_box and y_box
      x_box = idx%grid_x
      y_box = idx/grid_x
      
      #generate the mask
      mask = np.full((rows,cols),255).astype(np.uint8)
      mask[y_box*y_subsec:(y_box+1)*y_subsec,x_box*x_subsec:(x_box+1)*x_subsec] = 0

      # occlude image
      occ_img = cv2.bitwise_and(img,img,mask=mask)

      #append occluded image to container
      occluded_images.append(occ_img)

  return occluded_images

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
    '--horiz_stretch',
    nargs='*',
    type=float,
    help='List that contains [n_stretch,max_stretch]. '+
    'Stretches data n_stretch times horizontally, with max_stretch as limit. '+
    'The limit is defined as a scale of the original image {0.01;100} '+
    '(max>1 is stretch, max<1 is compression).'
  )
  parser.add_argument(
    '--vert_stretch',
    nargs='*',
    type=float,
    help='List that contains [n_stretch,max_stretch]. '+
    'Stretches data n_stretch times vertically, with max_stretch as limit. '+
    'The limit is defined as a scale of the original image {0.01;100} '+
    '(max>1 is stretch, max<1 is compression).'
  )
  parser.add_argument(
    '--horiz_shear',
    nargs='*',
    type=float,
    help='List that contains [n_shear,max_shear]. '+
    'Shears data n_shear times horizontally, with max_shear as limit. '+
    'The limit is defined as a portion of the original image to move -> {0;10}'
  )
  parser.add_argument(
    '--vert_shear',
    nargs='*',
    type=float,
    help='List that contains [n_shear,max_shear]. '+
    'Shears data n_shear times vertically, with max_shear as limit. '+
    'The limit is defined as a portion of the original image to move -> {0;10}'
  )
  parser.add_argument(
    '--gaussian_noise',
    nargs='*',
    type=float,
    help='List that contains [mean,std] for the Gaussian noise applied'
  )
  parser.add_argument(
    '--occlude',
    nargs='*',
    type=int,
    help='List that contains [grid_x,grid_y,selection] for the occlusions '+ 
    'applied. Grid_x and Grid_y set the x and y amounts for the grid, and '+
    'selection is a list that contains which section\'s occlusions we want'+
    'to apply. See source code for explanation.'
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

  # horizontal and vertical stretch sanity check. Limiting to 100 stretches, 
  # and limiting max and min stretch to 0.01 and 100, since at that point it's
  # already mom's spaghetti and probably a mistake
  if FLAGS.horiz_stretch and (len(FLAGS.horiz_stretch) != 2):
    print("Wrong usage of horizontal stretch parameters. Check again. Exiting")
    quit()

  n_horiz_stretch = 0
  if FLAGS.horiz_stretch:
    n_horiz_stretch = int(FLAGS.horiz_stretch[0])
    max_horiz_stretch = FLAGS.horiz_stretch[1]
    if n_horiz_stretch > 100:
      print("Too many horizontal stretches. Exiting")
      quit()
    if max_horiz_stretch > 100 or max_horiz_stretch < 0.01:
      print("Horizontal stretch size off boundaries. Exiting")
      quit()

  if FLAGS.vert_stretch and (len(FLAGS.vert_stretch) != 2):
    print("Wrong usage of vertical stretch parameters. Check again. Exiting")
    quit()

  n_vert_stretch = 0
  if FLAGS.vert_stretch:
    n_vert_stretch = int(FLAGS.vert_stretch[0])
    max_vert_stretch = FLAGS.vert_stretch[1]
    if n_vert_stretch > 100:
      print("Too many vertical stretches. Exiting")
      quit()
    if max_vert_stretch > 100 or max_vert_stretch < 0.01:
      print("Vertical stretch size off boundaries. Exiting")
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

  # occlusions sanity check
  if FLAGS.occlude:
    if(len(FLAGS.occlude) < 3):
      print("Wrong usage of occlusion parameters. Check again. Exiting")
      quit()
    
    if(FLAGS.occlude[0] < 0 or FLAGS.occlude[1] < 0):
      print("Wrong usage of occlusion grid parameters. Check again. Exiting")
      quit()
    else:
      x_grid=FLAGS.occlude[0]
      y_grid=FLAGS.occlude[1]

    for i in FLAGS.occlude[2:]:
      if(i<0 or i>=x_grid*y_grid):
        print("Wrong usage of occlusion selection. Off boundaries! Exiting")
        quit()
    occlusion_selection = FLAGS.occlude[2:]

  # parameter show.   
  print("----------------------------Parameters-------------------------------")
  print("in_dir: ",FLAGS.in_dir)
  print("in_im: ",FLAGS.in_dir)
  print("out_dir: ",FLAGS.out_dir)
  print("rots: ",FLAGS.rots)
  print("horiz_shear: ",FLAGS.horiz_shear)
  print("vert_shear: ",FLAGS.vert_shear)
  print("gaussian_noise",FLAGS.gaussian_noise)
  print("occlude",FLAGS.occlude)
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
  if n_horiz_stretch:
    print("Stretching images horizontally %d times, with max_stretch:%.2f" 
        % (n_horiz_stretch, max_horiz_stretch))
    horiz_stretch_list = apply_horiz_stretch(images,n_horiz_stretch,max_horiz_stretch)
    transformed_list.extend(horiz_stretch_list)
    print("Done!")
  if n_vert_stretch:
    print("Stretching images vertically %d times, with max_stretch:%.2f" 
        % (n_vert_stretch, max_vert_stretch))
    vert_stretch_list = apply_vert_stretch(images,n_vert_stretch,max_vert_stretch)
    transformed_list.extend(vert_stretch_list)
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
  if FLAGS.occlude:
    print("Applying occlusions with x_grid:%d, y_grid:%d"%(x_grid,y_grid))
    occ_images = apply_occlusions(images,x_grid,y_grid,occlusion_selection)
    transformed_list.extend(occ_images)
    print("Done!")
  #if FLAGS.lalalala... Other transformations

  # @TODO: Save to disk in out_dir!!!!!!!!!!!!!!!

  # if asked, show the results
  if FLAGS.show:
    for img in transformed_list:
      shim.im_plt(img)
    shim.im_block()


