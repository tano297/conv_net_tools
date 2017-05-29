#!/usr/bin/python2.7

"""
Quick script to show an image using opencv and matplotlib because I never
remember how it is and I am too lazy to google.
"""
import argparse
import matplotlib.pyplot as plt
import cv2

def cv2_plt(img):
  """
    Open image and print it on screen
  """
  plt.ion()
  plt.figure()
  img = cv2.imread(img)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Takes input image and plots it")
  parser.add_argument(
    'img',
    type=str,
    help='Path for image. No default'
  )
  FLAGS, unparsed = parser.parse_known_args()

  #plot the image
  print("Opening image \'%s\'"%FLAGS.img)
  cv2_plt(FLAGS.img)
  cv2_plt(FLAGS.img)
  plt.show(block=True)
  

