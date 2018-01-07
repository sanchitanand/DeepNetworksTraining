

Deep Neural Network Training - A program written using python Tensorflow package to train a neural network 
six hidden layers, for the purpose of classifying fMRI data in a kaggle contest (kaggle.com/c/cs446-fa17). 
The data is available via google drive: https://drive.google.com/file/d/1c_iTWbBGoYPfI7-iVfoKmPDl3jGhJYWg/view?usp=sharing
The data is organized as follows:

train_X.npy - a numpy array consisting of 4602 training samples of 3D fMRI brainscans. The array dimensions are 4602x26x31x23
tag_name.npy - names of 19 different labels corresponding to distinct brain activities
train_binary_Y.npy - labels for each of the 4602 training samples. the array dimensions are 4602x19  
valid_test_X.npy - array of 1971 unclassified samples, for the purpose of competition testing. The array dimensions are 1971x26x31x23

The neural network uses ReLu activation for hidden nodes, and sigmoid for output node. Each hidden layer is 1000 nodes wide. 
The hyperparameters were tuned after extensive testing.  The neural network makes use of normalization, regularization and dropout to 
prevent overfitting. Batch size was fixed at 200. The network trains achieves good results after 200-300 epochs. 
	
Copyright (c) 2018 Sanchit Anand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


