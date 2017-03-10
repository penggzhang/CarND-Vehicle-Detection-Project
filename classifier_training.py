import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time, pickle
from helper_functions import *


# Call helper function to read cars and notcars data
cars, notcars = read_data()
print("Number of cars: ", len(cars))
print("Number of not cars: ", len(notcars))


# Reduce the sample size to test the training pipeline
#sample_size = 500
#rand_ind = np.random.randint(0, len(cars), sample_size)
#cars = np.array(cars)[rand_ind]
#rand_ind = np.random.randint(0, len(notcars), sample_size)
#notcars = np.array(notcars)[rand_ind]
#print("\nSample size: ", sample_size)


# Set parameters for feature extraction
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32 # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off


# Extract features
print("\nExtracting features ...")
t=time.time()
# Call helper function to extract features
car_features = extract_features(cars, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    orient=orient, pix_per_cell=pix_per_cell, 
                    cell_per_block=cell_per_block, hog_channel=hog_channel, 
                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    orient=orient, pix_per_cell=pix_per_cell, 
                    cell_per_block=cell_per_block, hog_channel=hog_channel, 
                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'seconds to extract features.')


# Prepare datasets
# Stack features
print("\nStacking features ...")
t=time.time()
X = np.vstack((car_features, notcar_features)).astype(np.float64)
t2 = time.time()
print(round(t2-t, 2), 'seconds to stacking features.')

# Normalize features
print("\nNormalizing features ...")
t=time.time()
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
t2 = time.time()
print(round(t2-t, 2), 'seconds to normalizing features.')

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
print("\nSplitting training and test datasets ...")
t=time.time()
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                        test_size=0.2, random_state=rand_state)
t2 = time.time()
print(round(t2-t, 2), 'seconds to splitting datasets.')

# Print information about feature extraction
print('\nUsing:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block',
      'in', color_space, 'with', hog_channel, 'channel(s)')
print('Feature vector length:', len(X_train[0]))


# Use a linear SVC 
svc = LinearSVC()
print("\nTraining the classifier ...")
t=time.time()
# Train the classifier
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'seconds to train SVC.')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


# Check prediction for a few samples
print("\nPredicting with the trained classifier ...")
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'seconds to predict', n_predict,'labels with SVC.')


# Save the trained classifier, scaler and parameters for feature extraction
model_data = (svc, X_scaler, color_space,
              orient, pix_per_cell, cell_per_block, hog_channel,
              spatial_size, hist_bins, 
              spatial_feat, hist_feat, hog_feat)
model_file = "svc.p"
save_model(model_data, model_file)


















