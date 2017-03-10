##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test\_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_image.jpg
[image2]: ./output_images/search_and_classify.jpg
[image3]: ./output_images/hog_subsampling.jpg
[image4]: ./output_images/heatmap_and_bbox.jpg
[image5]: ./output_images/frames_and_heatmaps.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/last_frame_with_bbox.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here the rubric points are considered individually and are described how they are addressed in implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the helper function of `get_hog_features()` as the lines `43~60` of the file `helper_functions.py`. This function works for HOG feature extraction for single color channel.

A classifier training was started by reading in all the `vehicle` and `non-vehicle` images as the lines `10~12` of `classifier_training.py`. Then different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, `cells_per_block` and `hog_channel`) were explored as the lines `25~34` of `classifier_training.py`. 

To get a feel for what the HOG feature and HOG image output looks like, some random images were grabbed from each of the two classes and displayed as the code in lines `9~58` of `detecting.py`. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `hog_channel=0`:

![alt text][image1]

The helper function `single_img_features()` calls `get_hog_features()` to get the HOG features, as seen in lines `65~118` of `helper_functions.py`.

####2. Explain how you settled on your final choice of HOG parameters.

In order to settle on the choice of HOG parameters, a classifier training pipeline, as the file `classifier_training.py`, was implemented, and various combinations of parameters for feature extraction, as the lines `25~34` of the file, were tried.

Eventually, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `hog_channel="ALL"` were found as the best HOG parameters since they brought the highest test accuracy for the classification. `YCrCb` was also found the best color space working with the above HOG parameters for feature extraction.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A linear SVM was trained using the classifier `sklearn.svm.LinearSVC()` and the training dataset provided in the project repo. The code of this step is contained the file `classifier_training.py`.

Helper function `extract_features()` were called, as the lines `40~50` of `classifier_training.py`, to extract features for a list of images. The selected HOG parameters as above as well as other color-based feature parameters (`spatial_size=(32, 32)` and `hist_bins=32`) were passed into this function, and then corresponding HOG (shape-based) and color-based features were extracted. The code of `extract_features()` is contained in lines `121~171` of `helper_functions.py`, and the code for spatial binning `bin_spatial()` and color histogram `color_hist()` feature extraction is also contained in the same file as lines `25~40`.

Back to the classifier training pipeline in `classifier_training.py`. The extracted features were then stacked into the X dataset, and then was normailzed by `sklearn.preprocessing.StandardScaler()`. Label vector y was also defined. Then the whole data set was splitted into randomized training and test sets by calling `sklearn.model_selection.train_test_split()`.

Then the linear SVC was fit on the data. Test accuracy was reported, and parameters for feature extraction were tuned correspondingly as mentioned above.

At the end of the training, the trained classifier, feature scaler and parameters for feature extraction were stored into a pickle file.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

In general, two kinds of sliding window search methods were used to detect where the vehicle is in images. First one was implemented by functions `slide_window()` and `search_windows()` which are contained in the lines `239~307` of `helper_functions.py`. Second one was implemented by functions `find_cars()` and `detect()` which are contained in lines `311~437` of `helper_functions.py`. Both of them find the vehicle-in boxes in images.

Using the first method, function `search_classify()` as lines `61~123` in `detecting.py` was called to conduct the detection pipeline work on test images. The trained classifier and all other parameters in file of `svc.p`  were loaded. Search area in images `y_start_stop=[400, 656]`, the size `xy_window=(96, 96)` and overlap ratio `overlap=0.5` of sliding window were tried and defined. After finding all windows to be searched through `slide_window()`, classification was done on each window patch through `search_window()`, and then the window boxes with vehicle in were returned.

Here is the output on test images with the found boxes drawn on:

![alt text][image2] 

The second method applies a so-called HOG sub-sampling technique and extracts HOG features only once, which is thus more efficient than the first method. The function `hog_subsampling_detect()` as lines `126~166` of `detecting.py` was called to execute a pipeline of search and classification based on this method. The trained classifier and stored parameters were passed into the function `detect()`, in which multi-scale windows defined as `scale_list=[1.5, 2]` are allowed for search. Then the underlying `find_cars()` function was called to use HOG sub-sampling technique to extract features for each window and return the classification result, i.e. boxes with vehicle found in. 

Note that here search window is sliding at a pace defined by another parameter `cells_per_step` instead of the `overlap` of first method. After trial, 2 `cells_per_step` were selected. The search area in images was again set as `y_start=400` and `y_stop=656` to ignore the tree top region where vehicle are not possible to appear. Also, the scales of 1.5 and 2, sampling ratio over images, were found the best ones for the pipeline task. 

Here is the output on test images using HOG sub-sampling method:

![alt text][image3]

Following the detection of all vehicle-in boxes, the pipeline went on as the function `draw_heatmap_bbox()` in the lines `169~218` of `detecting.py`. A heat map was made for each processed image from those hot boxes by calling the `add_heat()` function as the lines `451~459` of `helper_functions.py`. And then the heat map was labeled by the `scipy.ndimage.measurements.label()` function. Given such labels, bounding boxes where vehicle were labeled were drawn on by calling the function `draw_boundging_boxes()` as lines `462~475` of `helper_functions.py`. 

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately the search is performed on two scales, 1.5 and 2, using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are the output on test images with bounding boxes draw on and heat maps shown on the right:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A class named as `tracker` in the file `tracker.py` was defined to record all vehicle-in boxes that were detected from a certain number of video frames. By calling the tracker's `track()` method, a sequence of heat maps were created one by one based on the hot boxes from each frame and stored by the class's variable `heatmap_list`. The number of frames was defined by `frame_factor`, another variable of the class.

Thresholding on single heat map might be applied to remove the blobs under a threshold. However, single heat map threshold was eventually found not quite useful and was thus set as 0. 

Then a series of heat maps were sum up element-wise into an integrated heat map, and class's method `threshold_multiple_heatmap()` was called on this integrated heat map to remove the blobs under the `multiple_heat_thresh`. Such method was proved to be effective to combine multiple detections and reject false positives. Then `scipy.ndimage.measurements.label()` was called to identify individual blobs in the integrated heatmap and identify the vehicle position. 

The resulting labels were then returned to its caller in line `17` of `main.py`, and corresponding bounding boxes covering the area of each detected vehicle were drawn on the image frame. Such implementation was wrapped up into a main pipeline function `process_image()`.

Also in `main.py`, the `model_file` stored with trained classifier and parameters was set, search position of `ystart` and `ystop` in images was set, image sampling `scale_list` was set, an instance of `tracker` was initiated with `frame_factor` and `multiple_heat_thresh` set as `10` and `7` respectively, as well as the video to be processed was set.

The `process_image()` function was passed into video clip's method `fl_image()` along with all the above settings as global variables. And then an output video labeled with vehicle-in bounding boxes was obtained.
 
Below is an example result showing the heatmap from a series of frames of test video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video. The code of this step is contained in `integrate_heatmaps()` as lines `221~286` of `detecting.py`.

### Here are 6 frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 6 frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems / issues:

1. At the very first moment a vehicle coming into the vision (image) from the edge, it cannot be correctly identified.
2. When two cars get close to each other, they merge into one bounding box.
3. The bounding boxes are still wobbly.
4. There is a bit detection across guardrail on the other side of highway.

The current pipeline is likely to fail where the first 2 problems or issues as above happened. 

To make it more robust, the following improvement shall be done:

1. Train classifier further using better or augmented data, like images of part of vehicle. Make it able to correctly classify vehicles when they are appearing into the vision range.
2. Write more scripts to track individual vehicle, like creating vehicle instance for each correct detection and keeping trace of them to avoid incorrectly merge multiple vehicle together.
3. Average the positions and sizes of bounding boxes over frames. By this way, the wobbly bounding boxes shall be smoothed out.





















 

