import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle, cv2, glob, time
from scipy.ndimage.measurements import label
from helper_functions import *


def preview_hog_image():
	# 1) Load random samples from the data set;
	# 2) Extract HOG features for those images;
	# 3) Plot the images and their HOG counterparts.

	# Call helper function to load data
	cars, notcars = read_data()

	# Choose random car / not-car index
	car_ind = np.random.randint(0, len(cars))
	notcar_ind = np.random.randint(0, len(notcars))

	# Read in car / not-car images
	car_img = mpimg.imread(cars[car_ind])
	notcar_img = mpimg.imread(notcars[notcar_ind])

	# Define parameters for feature extraction
	color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channel = 0 # Can be 0, 1, 2, or "ALL" # Note: Use single channel for preview
	spatial_size = (32, 32) # Spatial binning dimensions
	hist_bins = 32    # Number of histogram bins
	spatial_feat = True # Spatial features on or off
	hist_feat = True # Histogram features on or off
	hog_feat = True # HOG features on or off

	# Extract HOG features and HOG image
	car_features, car_hog_img = single_img_features(car_img, color_space=color_space, 
                    				spatial_size=spatial_size, hist_bins=hist_bins, 
                    				orient=orient, pix_per_cell=pix_per_cell, 
                    				cell_per_block=cell_per_block, hog_channel=hog_channel, 
                    				spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                    				vis=True)
	notcar_features, notcar_hog_img = single_img_features(notcar_img, color_space=color_space, 
                    				spatial_size=spatial_size, hist_bins=hist_bins, 
                    				orient=orient, pix_per_cell=pix_per_cell, 
                    				cell_per_block=cell_per_block, hog_channel=hog_channel, 
                    				spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                        			vis=True)

	# Plot
	images = [car_img, car_hog_img, notcar_img, notcar_hog_img]
	titles = ["car image", "car HOG image", "notcar image", "notcar HOG image"]
	fig = plt.figure(figsize=(12,3))
	visualize(fig, 1, 4, images, titles)
	#fig.savefig("output_images/hog_image.jpg")

#preview_hog_image()


def search_classify():
	# 1) Load the trained classifier and other parameters;
	# 2) Set search position, window size and overlap;
	# 3) Find all windows to search;
	# 4) Perform classification on the features for each window,
	#    and thus search out if any vehicle exists in each window;
	# 5) Draw boxes on the windows where vehicle is found;
	# 6) Plot to verify that the search and the classifier works well.

	# Load the trained classifier, scaler and parameters
	# for feature extraction
	model_file = "svc.p"
	(svc, X_scaler, color_space,
	 orient, pix_per_cell, cell_per_block, hog_channel,
	 spatial_size, hist_bins, 
	 spatial_feat, hist_feat, hog_feat) = load_model(model_file)

	# Set the start and stop position of x and y 
	# in image for search
	x_start_stop = [None, None]
	y_start_stop = [400, 656]
	# Set the size and overlap of searching window
	xy_window = (96, 96)
	overlap = 0.5 

	# Prepare output lists for box-drawn-on images and titles
	images, titles = [], []
	# Step through test images
	fnames = glob.glob("test_images/*.jpg")
	for fname in fnames:
		# Check the time for processing one individual image
		t = time.time()
		# Read in image and prepare an image copy for drawing
		img = mpimg.imread(fname)
		draw_img = np.copy(img)
		# Note: Divide image arrays by 255 because training images are png files, 
		# and matplotlib.image.imread() reads them as [0~1].
		img = img.astype(np.float32)/255 
		# Call helper function to find all windows to search
		windows = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
					xy_window=xy_window, xy_overlap=(overlap, overlap))
		# Call helper function to search each window, apply the trained classifier
		# on features of this window and return any hot one if vehicle is found in it.
		# The stored scaler and parameters for feature extraction are also applied.
		hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, 
						spatial_size=spatial_size, hist_bins=hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, 
						cell_per_block=cell_per_block, hog_channel=hog_channel, 
						spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
		# Call helper function to draw boxes on these hot windows
		window_img = draw_boxes(draw_img, hot_windows)
		# Append the box-drawn-on image to output lists
		images.append(window_img)
		titles.append(fname.split('/')[-1][:-4])
		# Print out processing time and the number of windows that were searched
		print(time.time()-t, "seconds to search one image with", len(windows), "windows.")

	# Plot
	fig = plt.figure(figsize=(12,6))
	visualize(fig, 3, 2, images, titles)
	#fig.savefig("output_images/search_and_classify.jpg")

#search_classify()


def hog_subsampling_detect():
	# 1) Set the model file to load, search position and image sampling scale;
	# 2) Search and classify using HOG sub-sampling, by which HOG features
	#    are extracted only once. Obtain vehicle-in boxes;
	# 3) Draw boxes on the vehicle-in boxes;
	# 4) Plot to verify that HOG sub-sampling works well.

	# Set the model file to load the trained classifier and other parameters
	model_file = "svc.p"
	# Set the start and stop position of y for search
	ystart = 400
	ystop = 656
	# Set scale for image sampling. Multi-scales are allowed.
	scale_list = [1.5, 2]

	# Step through test images
	images, titles = [], []
	fnames = glob.glob("test_images/*.jpg")
	for fname in fnames:
		# Check the time for processing one individual image
		t = time.time()
		# Read in image and prepare an image copy for drawing
		img = mpimg.imread(fname)
		draw_img = np.copy(img)
		# Call helper function to search and classify by HOG sub-sampling.
		# Obtain a list of boxes that contain vehicle.
		boxes = detect(img, model_file, ystart, ystop, scale_list)
		# Call helper function to draw boxes on
		output_img = draw_boxes(draw_img, boxes)
		# Append the box-drawn-on image to output lists
		images.append(output_img)
		titles.append(fname.split('/')[-1][:-4])
		# Print out processing time
		print(time.time()-t, "seconds to detect one image.")

	# Plot
	fig = plt.figure(figsize=(12, 6))
	visualize(fig, 3, 2, images, titles)
	#fig.savefig("output_images/hog_subsampling.jpg")

#hog_subsampling_detect()


def draw_heatmap_bbox():
	# 1) Set the model file to load, search position and image sampling scale;
	# 2) Search and classify to obtain vehicle-in boxes;
	# 3) Make a heat map given the vehicle-in boxes;
	# 4) Label the heat map;
	# 5) Draw bounding box where vehicle is labeled;
	# 6) Plot to verify that heat maps and bounding boxes are done well.

	# Set the model file to load the trained classifier and other parameters
	model_file = "svc.p"
	# Set the start and stop position of y for search
	ystart = 400
	ystop = 656
	# Set scale for image sampling. Multi-scales are allowed.
	scale_list = [1.5, 2]

	# Step through test images
	images, titles = [], []
	fnames = glob.glob("test_images/*.jpg")
	for fname in fnames:
		# Check the time for processing one individual image
		t = time.time()
		# Read in image
		img = mpimg.imread(fname)
		# Call helper function to search and classify by HOG sub-sampling.
		# Obtain a list of boxes that contain vehicle.
		boxes = detect(img, model_file, ystart, ystop, scale_list)
		# Call helper function to make a heat map given the vehicle-in boxes
		heatmap = add_heat(img, boxes)
		# Call scipy function to label the heat map
		labels = label(heatmap)
		# Draw bounding box(es) where vehicle is labeled
		if labels[1] >= 1:
			draw_img = draw_bounding_boxes(img, labels)
		else:
			draw_img = np.copy(img)
		# Append the bounding-box-drawn-on image and heat map to output lists
		images.append(draw_img)
		images.append(heatmap)
		titles.append(fname.split('/')[-1][:-4])
		titles.append(fname.split('/')[-1][:-4])
		# Print out processing time
		print(time.time()-t, "seconds to process one image.")

	# Plot
	fig = plt.figure(figsize=(24, 12))
	visualize(fig, 6, 2, images, titles)
	#fig.savefig("output_images/heatmap_and_bbox.jpg")

#draw_heatmap_bbox()


def integrate_heatmaps():
	# 1) Detect vehicle-in boxes in a sequence of frames from the test video
	# 2) Create heat map for each frame given the detected hot boxes
	# 3) Integrate the sequence of heat maps
	# 4) Apply multiple heat thresholding on the integrated heat map
	# 5) Label the thresholded heat map
	# 6) Draw bounding boxes onto the last frame given the labels, 
	#    and thus identify the vehicle position

	# Load model file
	model_file = "svc.p"
	# Set y start and stop position for search in image
	ystart = 400
	ystop = 656
	# Set scale for image sampling. Multi-scales are allowed.
	scale_list = [1.5, 2]
	# Set multiple heat threshold
	multiple_heat_thresh = 5

	# Prepare output image list for displaying
	images, titles = [], []
	# Get all frame filenames into a list
	fnames = glob.glob("test_video_frames/*.jpg")
	# Create a zero heat map for integrating the multiple maps
	res_heatmap = np.zeros_like(mpimg.imread(fnames[0])[:,:,0]).astype(np.float)
	# Step through a series of frames
	for fname in fnames:
		img = mpimg.imread(fname)
		images.append(img)
		title = fname.split('/')[-1][-4]
		titles.append(title)
		# Detect vehicle-in boxes in single image frame
		box_list = detect(img, model_file, ystart, ystop, scale_list)
		# Create heat map
		heatmap = add_heat(img, box_list)
		images.append(heatmap)
		titles.append(title + "heat map")
		# Sum up (integrate) heat maps
		res_heatmap += heatmap

	# Plot frames and corresponding heat maps
	fig_1 = plt.figure(figsize=(24, 12))
	visualize(fig_1, 6, 2, images, titles)
	#fig_1.savefig("output_images/frames_and_heatmaps.jpg")

	# Threshold the integrated heat map
	res_heatmap[res_heatmap <= multiple_heat_thresh] = 0
	res_heatmap = np.clip(res_heatmap, 0, 255)
	# Label the thresholded heat map
	labels = label(res_heatmap)
	# Plot label image
	fig_2 = plt.figure()
	plt.imshow(labels[0], cmap='gray')
	plt.show()
	#fig_2.savefig("output_images/labels_map.jpg")

	# Draw bounding boxes onto the last frame
	last_frame = mpimg.imread(fnames[-1])
	draw_img = draw_bounding_boxes(last_frame, labels)
	# Plot the image with bounding box on
	fig_3 = plt.figure()
	plt.imshow(draw_img)
	plt.show()
	#fig_3.savefig("output_images/last_frame_with_bbox.jpg")

#integrate_heatmaps()


















