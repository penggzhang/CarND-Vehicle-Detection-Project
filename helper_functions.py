import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, pickle
from skimage.feature import hog

# Read file names from vehicles and non-vehicles directories
# into two lists, cars and notcars, respectively.
def read_data():
    cars = [fname for fname in glob.glob('vehicles/**/*.png', recursive=True)]
    notcars = [fname for fname in glob.glob('non-vehicles/**/*.png', recursive=True)]
    return cars, notcars

# Convert color space
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Extract spatial binning features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# Extract color histogram features                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Extracting HOG features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Extract features for a single image (or window patch).
# This function is very similar to extract_features()
# just for a single image rather than list of images.
def single_img_features(img, color_space='RGB', 
                        spatial_size=(32, 32), hist_bins=32, 
                        orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis=False):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))     
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], 
                                          orient, pix_per_cell, cell_per_block, 
                                          vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                               orient, pix_per_cell, cell_per_block, 
                               vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)
    #9) Return concatenated array of features
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)

# Extract features for a list of images
def extract_features(imgs, color_space='RGB', 
                     spatial_size=(32, 32), hist_bins=32, 
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []   
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        # Extract spatial binning features if flag is set
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        # Extract color histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        # Extract HOG features
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        # Concatenate and append the features of this file to list
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Plot multiple images
def visualize(fig, rows, cols, images, titles):
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i], fontsize=10)
        else:
            plt.imshow(img)
            plt.title(titles[i], fontsize=10)
    plt.tight_layout()
    plt.show()

# Save the trained classifier, scaler and parameters for feature extraction
# into a pickle file
def save_model(model_data, model_file):
    (svc, X_scaler, color_space,
     orient, pix_per_cell, cell_per_block, hog_channel,
     spatial_size, hist_bins, 
     spatial_feat, hist_feat, hog_feat) = model_data

    dict_pickle = {}
    dict_pickle["svc"] = svc
    dict_pickle["scaler"] = X_scaler
    dict_pickle["color_space"] = color_space
    dict_pickle["orient"] = orient
    dict_pickle["pix_per_cell"] = pix_per_cell
    dict_pickle["cell_per_block"] = cell_per_block
    dict_pickle["hog_channel"] = hog_channel
    dict_pickle["spatial_size"] = spatial_size
    dict_pickle["hist_bins"] = hist_bins
    dict_pickle["spatial_feat"] = spatial_feat
    dict_pickle["hist_feat"] = hist_feat
    dict_pickle["hog_feat"] = hog_feat

    with open(model_file, "wb") as f:
        pickle.dump(dict_pickle, f)
    print("\nClassifier saved.\n")

# Load the trained classifier, scaler and parameters for feature extraction
# from a pickle file
def load_model(model_file):
    dict_pickle = pickle.load(open(model_file, "rb" ))

    svc = dict_pickle["svc"]
    X_scaler = dict_pickle["scaler"]
    color_space = dict_pickle["color_space"]
    orient = dict_pickle["orient"]
    pix_per_cell = dict_pickle["pix_per_cell"]
    cell_per_block = dict_pickle["cell_per_block"]
    hog_channel = dict_pickle["hog_channel"]
    spatial_size = dict_pickle["spatial_size"]
    hist_bins = dict_pickle["hist_bins"]
    spatial_feat = dict_pickle["spatial_feat"]
    hist_feat = dict_pickle["hist_feat"]
    hog_feat = dict_pickle["hog_feat"]

    return (svc, X_scaler, color_space,
            orient, pix_per_cell, cell_per_block, hog_channel,
            spatial_size, hist_bins, 
            spatial_feat, hist_feat, hog_feat)

# Function that takes an image, start and stop positions in both x and y, 
# window size (x and y dimensions), and overlap fraction (for both x and y),
# and returns all windows to be searched.
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Function that takes an image and the list of windows 
# to be searched (output of slide_windows()), and returns
# all vehicle-in windows
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                   spatial_size=(32, 32), hist_bins=32,
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, 
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    orient=orient, pix_per_cell=pix_per_cell, 
                    cell_per_block=cell_per_block, hog_channel=hog_channel, 
                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Extract features using hog sub-sampling and make predictions.
# Return a list of vehicle-in boxes.
def find_cars(img, ystart, ystop, scale, svc, X_scaler, 
              orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              spatial_feat, hist_feat, hog_feat):
    
    #draw_img = np.copy(img)

    # Note: Divide image arrays by 255 because training images are png files, 
    # and matplotlib.image.imread() reads them as [0~1].
    img = img.astype(np.float32)/255

    # Filter out the area of image to be searched
    img_tosearch = img[ystart:ystop,:,:]
    # Convert color space
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    # Sample the image to the given scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        new_size = (np.int(imshape[1]/scale), np.int(imshape[0]/scale))
        ctrans_tosearch = cv2.resize(ctrans_tosearch, new_size)

    # Separate each channel
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps
    nxcells = (ch1.shape[1] // pix_per_cell)
    nycells = (ch1.shape[0] // pix_per_cell)
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    cell_per_window = (window // pix_per_cell)
    # Instead of overlap, define how many cells to step
    cells_per_step = 2
    # Window moving steps
    nxsteps = (nxcells - cell_per_window) // cells_per_step
    nysteps = (nycells - cell_per_window) // cells_per_step
    # Calculate how many blocks in one window
    block_per_window = cell_per_window - cell_per_block + 1

    # Compute individual channel HOG features for the entire image
    # Set feature_vec as False, so that return HOG feature array 
    # with first two dimensions are block indices
    if hog_feat == True:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Prepare a list to store boxes that are found with vehicle inside
    box_list = []

    # Slide window step by step and search
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            # Prepare a list for features of this window patch
            window_features = []

            # Window position given by cells
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Window position given by pixels
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Extract spatial binning features
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                window_features.append(spatial_features)

            # Extract color histogram features
            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                window_features.append(hist_features)

            # Extract HOG features by sub-sampling blocks for this window patch
            if hog_feat == True:
                hog_feat1 = hog1[ypos:ypos+block_per_window, xpos:xpos+block_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+block_per_window, xpos:xpos+block_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+block_per_window, xpos:xpos+block_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                window_features.append(hog_features)

            # Integrate features for this window patch
            window_features = np.concatenate(window_features).reshape(1, -1)

            # Scale features
            test_features = X_scaler.transform(window_features)

            # Make prediction
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                # Locate the box (window) position in the image of original scale
                box_x_left = np.int(xleft * scale)
                box_y_top = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                top_left_vertex  = (box_x_left, box_y_top+ystart)
                bot_right_vertex = (box_x_left+win_draw, box_y_top+win_draw+ystart)
                box = (top_left_vertex, bot_right_vertex)

                #cv2.rectangle(draw_img, box[0], box[1], (0,0,255), 6)

                # Append the found vehicle-in box to box_list
                box_list.append(box)

    # Return the list of vehicle-in boxes
    return box_list

# Take an image, the stored model file, search position for y
# and sampling scale for the image. Call find_cars function to obtain
# all vehicle-in boxes.
def detect(img, model_file, ystart, ystop, scale_list):
    # Get the stored classifier and features extraction parameters
    (svc, X_scaler, color_space,
     orient, pix_per_cell, cell_per_block, hog_channel,
     spatial_size, hist_bins, 
     spatial_feat, hist_feat, hog_feat) = load_model(model_file)
    # Prepare a list to store the found vehicle-in box lists
    multi_box_list = []
    # Multi-scales are allowed
    for scale in scale_list:
        box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, 
                             orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                             spatial_feat, hist_feat, hog_feat)
        multi_box_list.extend(box_list)
    return multi_box_list

# Draw boxes on image
def draw_boxes(img, boxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for box in boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, box[0], box[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Make heat map given a list of boxes
def add_heat(img, box_list):
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    # Iterate through list of boxes
    for box in box_list:
        # Add += 1 for all pixels inside each box
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
    return heatmap

# Draw bounding boxes given labels
def draw_bounding_boxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
















