# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./new_images/yield.png "Traffic Sign 7"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/orig_image.png "Original Image"
[image4]: ./new_images/70_km_h.png "Traffic Sign 1"
[image5]: ./new_images/road_works.png "Traffic Sign 2"
[image6]: ./new_images/road_works_1.png "Traffic Sign 3"
[image7]: ./new_images/stop_1.png "Traffic Sign 4"
[image8]: ./new_images/stop.png "Traffic Sign 5"
[image9]: ./examples/data_exploration.png "Data exploration"
[image10]: ./examples/histogram.png "Histogram"
[image11]: ./examples/rgbscale.png "RGB"
[image12]: ./examples/augmented.png "Augmented image"
[image13]: ./examples/aug_histogram.png "Augmented histogram"
[image14]: ./examples/lenet.png "Lenet"
[image15]: ./new_images/stop_spanish.png "Traffic Sign 6"
[image16]: ./new_images/yield_2.png "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

Here is a link to my [project code](https://github.com/vmonestel/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary statistics of the traffic signs data set using python:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image9]

This is the histogram that shows how the training set is distribuited among the classes. There are classes that have few samples (i.e class 0) and some that have a large amount of images (class 1).

![alt text][image10]

### Design and Test a Model Architecture

#### 1. Data preprocessing.

As a first step, I decided to convert the images to grayscale because I want to focus on the image pixels and not the colors. Each sign is unique so it is better to focus on the sign form and details to traing the neural network.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image11]

I decided to generate additional data because as shown in the histogram; there are classes with few samples, so the neural network would tend to identify better the classes with large samples but it would fail to identify new images of classes with few samples. Increasing the data will result in better predictions for those classes with few data. To add more data I decided to generate new images from the original training set. I looked for classes with less than 1500 samples and generate the difference to achieve at 1500 samples. So I copied the original images and generate new images with random effects (scaling, rotating, brightness change, tranlating). It resulted in a more robust data set, with different images that helped me to achieve a higher accuracy.

Here is an example of an original image and an augmented image:

![alt text][image3] ![alt text][image12]

The original training set has 34799 images and the size of the augmented training set + the original is 67380. Data augmentation is an easy technique and one of its advantages is that the new images don't have to be stored in hard disk, however it requires some time and resources (memory) to generate the augmented set; this could be a problem in case of a very large datasets. This is how the final histogram looks:

![alt text][image13]


#### 2. Model architecture.

My final model consisted of the following layers (Orignal LeNet-5)
![alt text][image14]


1. Layer 1. 5x5 convolution (32x32x1 in, 28x28x6 out)
2. ReLU Activation
3. 2x2 max pool (28x28x6 in, 14x14x6 out)
4. Layer 2. 5x5 convolution (14x14x6 in, 10x10x16 out)
5. ReLU Activation
6. 2x2 max pool (10x10x16 in, 5x5x16 out)
7. Flatten layer (5x5x6 in, 400 out)
8. Fully connect (400 in, 120 out)
9. ReLU Activation
10. Dropout layer (75% kept)
11. Fully connect (120 in, 84 out)
12. ReLU Activation
13. Dropout layer (75% kept)
14. Fully connect (84 in, 43 out)

#### 3. Training the model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an:

* Btach size = 128
* epochs = 50
* Learning rate = 0.0007
* Optimizer: Adam Optimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps.

Initially, with no modification of the data set and the original lab lenet (no dropout) I got around 88% of accuracy in the validation set. After converting the images to gray and adding 25% of drops in the LeNet model, I started to get a higher accuracy (~92-93%) which wasn't enough for the project requirements. After that, I investigated about data augmentation and I started to apply it; at the beggining I got around 94-95% of accuracy but I was curious why it wasn't a little better. After some debugging I found a bug in the data augmentation code. My code was creating new images just with samples 0 and 1 of the class. So for example, if the class size is 200, then the code created new images to have 1500 samples; however it took sample 0 and create a new image, it took sample 1 and created an image, then with 0, then 1, ... So at the end I had 1300 new images but created on just 2 original samples. After I found the bug, the process is: take sample 0, create the image; take sample 1, create the image; take sample 2 and create the image; sample 4 and create the image and so on until I get 1300 new images based on the 200 original samples.

My final model results were:
* Train Accuracy = 0.998
* Valid Accuracy = 0.964
* Test Set Accuracy = 0.937
 
My guess is that to achieve better results some changes to the original LeNet should be made (LeNet provided good results but not the best). It seems to me that there is not any other to get around the 100% validation accuracy. 

### Test a Model on New Images

#### 1. Choose new German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image15] 
![alt text][image7] ![alt text][image8] ![alt text][image1] ![alt text][image16] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


