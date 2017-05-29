#!/usr/bin/python2.7

import argparse
import os
from os import listdir
from os.path import isfile, join
import random
from shutil import rmtree
import math

def randomize(directory):
  """
  Returns a path and a list with the randomized filenames.
  """
  #get filenames
  files = [ f for f in listdir(directory) if isfile(join(directory,f))]

  #randomize
  random.shuffle(files)

  return directory,files

def divide(int_dir,files,out_dir,train_split,valid_split,test_split):
  """
  Puts the randomized data in the 
  """

  #create output dir if it doesn't exist, erase if it does
  if os.path.exists(out_dir):
    print("Removing dir")
    rmtree(out_dir)
  print("Creating dir")
  os.makedirs(out_dir)

  #create train, valid and test dirs
  os.makedirs(join(out_dir,"train"))
  os.makedirs(join(out_dir,"valid"))
  os.makedirs(join(out_dir,"test"))

  print("aaaa", len(files) * float(train_split) / 100.0)

  #get the sets according to split
  train_num = int(math.floor(len(files) * float(train_split) / 100.0))
  train_set = files[0:train_num]
  valid_num = int(math.floor(len(files) * float(valid_split) / 100.0))
  valid_set = files[train_num:train_num+valid_num]
  test_num = int(math.floor(len(files) * float(test_split) / 100.0))
  test_set = files[train_num+valid_num:train_num+valid_num+test_num]

  #debug
  print("total data: ",len(files))
  print("train data: ",train_num)
  print("valid data: ",valid_num)
  print("test data: ",test_num)
  print("sum: ",train_num+valid_num+test_num)

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
  print("Randomizing data in %s" % FLAGS.in_dir)
  in_dir,randfiles = randomize(FLAGS.in_dir)
  print("Done!")

  # put them in their folders
  print("Putting data in %s" % FLAGS.out_dir)
  out_dir = divide(FLAGS.in_dir,randfiles,FLAGS.out_dir,
                   FLAGS.split[0],FLAGS.split[1],FLAGS.split[2])
  print("Done!")

