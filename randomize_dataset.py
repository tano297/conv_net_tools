#!/usr/bin/python2.7

import argparse
import os
from os import listdir
from os.path import isfile, join
import random
from shutil import rmtree as sh_rmtree
from shutil import copy as sh_copy
import math

def randomize(directory):
  """
  Returns a path and a list with the randomized filenames.
  """
  #get filenames
  print("Getting file names from dir \'%s\'" % directory)
  files = [ f for f in listdir(directory) if isfile(join(directory,f))]

  #randomize
  print("Shuffling file names from dir \'%s\'" % directory)
  random.shuffle(files)

  print("Shuffled %d file names from dir \'%s\'" % (len(files),directory))
  return directory,files

def divide(in_dir,files,out_dir,train_split,valid_split,test_split):
  """
  Puts the randomized data in the 
  """

  #create output dir if it doesn't exist, erase if it does
  if os.path.exists(out_dir):
    print("Removing dir \'%s\'" % out_dir)
    sh_rmtree(out_dir)
  print("Creating dir \'%s\'" % out_dir)
  os.makedirs(out_dir)

  #create train, valid and test dirs
  print("Creating dir \'%s\'" % join(out_dir,"train"))
  os.makedirs(join(out_dir,"train"))
  print("Creating dir \'%s\'" % join(out_dir,"valid"))
  os.makedirs(join(out_dir,"valid"))
  print("Creating dir \'%s\'" % join(out_dir,"test"))
  os.makedirs(join(out_dir,"test"))

  #report amount of total data
  files_num = len(files)
  print("Dividing %d files" % files_num)

  #get the sets according to split, and copy to it's output dir
  
  # train data
  print("Moving shuffled train data to dir \'%s\'" % join(out_dir,"train"))
  train_num = int(math.floor(files_num * float(train_split) / 100.0))
  train_set = files[0:train_num]
  for file_name in train_set:
    full_file_name = os.path.join(in_dir, file_name)
    if (os.path.isfile(full_file_name)):
        sh_copy(full_file_name, join(out_dir,"train"))
  print("Moved %d/%d shuffled samples to dir \'%s\'" 
    % (train_num,files_num,join(out_dir,"train")))

  # validation data
  print("Moving shuffled validation data to dir \'%s\'" % join(out_dir,"valid"))
  valid_num = int(math.floor(files_num * float(valid_split) / 100.0))
  valid_set = files[train_num:train_num+valid_num]
  for file_name in valid_set:
    full_file_name = os.path.join(in_dir, file_name)
    if (os.path.isfile(full_file_name)):
        sh_copy(full_file_name, join(out_dir,"valid"))
  print("Moved %d/%d shuffled samples to dir \'%s\'" 
        % (valid_num,files_num,join(out_dir,"valid")))


  # test data
  print("Moving shuffled test data to dir \'%s\'" % join(out_dir,"test"))
  test_num = int(math.floor(files_num * float(test_split) / 100.0))
  test_set = files[train_num+valid_num:train_num+valid_num+test_num]
  for file_name in test_set:
    full_file_name = os.path.join(in_dir, file_name)
    if (os.path.isfile(full_file_name)):
        sh_copy(full_file_name, join(out_dir,"test"))
  print("Moved %d/%d shuffled samples to dir \'%s\'" 
        % (test_num,files_num,join(out_dir,"test")))

  # copy orphan samples from rest to training data (free data, wohoo!)
  if(train_num+valid_num+test_num < files_num):
    orphan_num = files_num - train_num - valid_num - test_num
    print("Copying %d orphans to training data (rounding)"%orphan_num)
    orphans = files[train_num+valid_num+test_num:]
    for file_name in orphans:
      full_file_name = os.path.join(in_dir, file_name)
      if (os.path.isfile(full_file_name)):
        sh_copy(full_file_name, join(out_dir,"train"))
    print("Moved %d/%d shuffled samples to dir \'%s\'" 
          % (orphan_num,files_num,join(out_dir,"train")))


  return out_dir

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Randomize data in a directory, and put " +
                    "them in another directory with the following "+
                    "structure:\n" + 
                    "   * dir/train\n"+
                    "   * dir/valid\n"+
                    "   * dir/test\n")
  parser.add_argument(
    '--in_dir',
    type=str,
    help='Path of raw dataset. No default'
  )
  parser.add_argument(
    '--out_dir',
    type=str,
    default='/tmp/out_dir',
    help='Path of raw dataset. Defaults to \'%(default)s\''
  )
  parser.add_argument(
    '--split',
    nargs='*',
    type=int,
    default=[70,15,15],
    help='Split of the dataset between training data, validation data, and '+
    'test data (in %% points). Can be a list of 3 summing to 100, or a list '+
    'of 2, where the the remaining percentage will be used for test. '+
    'Defaults to \'%(default)s\''
  )
  FLAGS, unparsed = parser.parse_known_args()

  # Sanity checks
  # Input directory needs to be provided
  if(not FLAGS.in_dir):
    print("Dataset directory needs to be provided. Exiting")
    quit()

  # It also needs to exist
  if not os.path.exists(FLAGS.in_dir):
    print("Input directory needs to exist, Einstein :) Exiting")
    quit()

  # Split must be consistent
  if(type(FLAGS.split) is not list or len(FLAGS.split)<2 or len(FLAGS.split)>3):
    print("Invalid split size. Exiting")
    quit() 

  # Split needs to sum to less than 100%
  if(sum(FLAGS.split)>100):
    print("Split needs to add to less than 100. Exiting")
    quit()    

  # If not enough info was provided, calculate the rest
  if(len(FLAGS.split)==2):
    FLAGS.split.append(100-FLAGS.split[0]-FLAGS.split[1])
   
  print("----------------------------Parameters-------------------------------")
  print("in_dir: ",FLAGS.in_dir)
  print("out_dir: ",FLAGS.out_dir)
  print("split: ",FLAGS.split)
  print("---------------------------------------------------------------------")

  # randomize the files
  print("\n\n- Randomizing data in dir \'%s\'" % FLAGS.in_dir)
  in_dir,randfiles = randomize(FLAGS.in_dir)
  print("Done!")

  # put them in their folders
  print("\n\n- Putting data in dir \'%s\'" % FLAGS.out_dir)
  out_dir = divide(FLAGS.in_dir,randfiles,FLAGS.out_dir,
                   FLAGS.split[0],FLAGS.split[1],FLAGS.split[2])
  print("Done!")

