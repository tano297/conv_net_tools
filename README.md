# Convolutional Neural Networks - Tools
Some scripts and tools to work with datasets and make my life easier when training deep convolutional neural nets

#### Scripts
* **_augment_data.py_**: Contains normal transformations, occlusions and selections performed to data when training deep convolutional neural networks. It can be used as a module, to call the transformations, or it can be used as a program, taking in a folder or an image, and either writing the results on disk or showing them on screen.
* **_show_img.py_**: Quickly print an opencv image, and keep open (I usually just import this module and call the plot function, instead of having to deal with the opencv/matplotlib stuff). It has sample app to try out.
* **_split_data.py_**: Splits a dataset from a raw image folder into training/validation/test, according to the desired split. It randomizes the data before splitting. It has sample app to try out.

#### Keywords
* Convolutional neural networks (CNN)
* Computer vision
* Machine learning
* Deep learning
