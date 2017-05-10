**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* For those first two steps requred normalization of features and randomization of a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[image1]: ./output_images/image1.png
[image2]: ./output_images/image2.png
[image3]: ./output_images/image3.png
[image4]: ./output_images/image4.png
[image5]: ./output_images/image5.png
[video1]: ./project_video_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

All code for this project stores in Vehicle-Detection.ipynb IPython Notebook.

Code consists of 2 main Pipelines:

* Lane Lines Detection - **cells 1 - 15**
* Vehicle Detection - **cells 16 - 22**

---
#### 0. Dataset

Dataset includes::

* 8792 images with cars
* 8968 images without cars


### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for this step is contained in the code cell #16 of the IPython notebook.

Reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Explore different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled with a stable set:

```
color_space    = 'YCrCb'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9          # HOG orientations
pix_per_cell   = 8          # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel    = 'ALL'      # Can be 0, 1, 2, or "ALL"
spatial_size   = (32, 32)   # Spatial binning dimensions
hist_bins      = 32         # Number of histogram bins
spatial_feat   = True       # Spatial features on or off
hist_feat      = True       # Histogram features on or off
hog_feat       = True       # HOG features on or off
```
Tried different color spaces, but YCrCb shown best results.

Increasing the ```orientation``` enhanced the accuarcy of the classifier, but increased computational time.


####3. Train a classifier using selected HOG features and color features.

The code for this step is contained in the code cell #18 of the IPython notebook.

The extracted features where fed to LinearSVC model of sklearn with default settings. 
The trained model had accuracy of 99.35% on test dataset.

The trained model and parameters used for training were saved to pickle file to be further used by vehicle detection pipeline.


### Sliding Window Search

#### 1. Sliding window search.

For higher coverage of potential detections the multi-scale window approach was used. It's prevents calculation of feature vectors for the complete image and thus helps in speeding up the process.

| Scale 1       | Scale 1       | Scale 1       |
|:-------------:|:-------------:|:-------------:|
| ystart = 380  | ystart = 400  | ystart = 500  |
| ystop = 480   | ystop = 600   | ystop = 700   |
| scale = 1     | scale = 1.5   | scale = 2.5   |

The figure below shows the multiple scales under consideration overlapped on image.

![alt text][image3]

#### 2. Examples of test images to demonstrate how pipeline is working.

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

I saved the positions of true detections in each frame of the video. From the true detections I created a heatmap and then thresholded that map to identify vehicle positions:

![alt text][image4]

Final step is to join 2 pipelines: Lane Line Detection And Vehicle Detection:

![alt text][image5]
---

### Video Implementation

Here's a [link to my video result on YouTube](https://www.youtube.com/watch?v=nrLscZvDLdo)

---

### Discussion

- Neural Network could show a higher precision.

- Pipeline may have problems in difficult lighting and illumination conditions.

- The multi-window search may be optimized further for better speed and accuracy.
