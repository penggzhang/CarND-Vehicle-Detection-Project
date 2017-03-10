import numpy as np
from scipy.ndimage.measurements import label
from helper_functions import *


# Define heat map tracker that takes a single image frame
# and a list of detected boxes from that image.
# Store a sequence of heat maps coming out from video frames. 
# Apply thresholding on the multiple heat maps to 
# merge multiple detections and remove false positives. 
# Return labels of bounding boxes for the found vehicles.
class tracker():
	def __init__(self, single_heat_thresh=2, frame_factor=15, multiple_heat_thresh=30):
		# Threshold for single heat map
		self.single_heat_thresh = single_heat_thresh
		# The number of multiple frames (heat maps) to process
		self.frame_factor = frame_factor
		# Threshold for multiple heat maps
		self.multiple_heat_thresh = multiple_heat_thresh
		# List to store heat maps
		self.heatmap_list = []
		# Labels of bounding box(es)
		self.labels = None

	def threshold_single_heatmap(self, heatmap):
		# Zero out pixels below the threshold for single heat map
		heatmap[heatmap <= self.single_heat_thresh] = 0
		# Return thresholded map
		return heatmap

	def threshold_multiple_heatmap(self):
		# Perfrom addition on multiple heat maps
		output_heatmap = np.copy(self.heatmap_list[0])
		for heatmap in self.heatmap_list[1:]:
			output_heatmap += heatmap
		# Zero out pixels below the threshold for multiple heat map
		output_heatmap[output_heatmap <= self.multiple_heat_thresh] = 0
		# Return thresholded map
		return np.clip(output_heatmap, 0, 255)

	def track(self, img, box_list):
		# Call helper function to make a heat map
		heatmap = add_heat(img, box_list)
		# Apply thresholding for this single image
		thresholded_heatmap = self.threshold_single_heatmap(heatmap)
		# Append the heat map to a list, which stores heat maps
		# up to the number of defined frame_factor.
		if len(self.heatmap_list) < self.frame_factor:
			self.heatmap_list.append(thresholded_heatmap)
		else:
			self.heatmap_list.pop(0)
			self.heatmap_list.append(thresholded_heatmap)
		# Once heat maps stored to the number of frame_factor,
		# apply thresholding for the multiple heat maps,
		# and then label the thresholed heat map and thus 
		# output the labels of bounding boxes.
		if len(self.heatmap_list) == self.frame_factor:
			output_heatmap = self.threshold_multiple_heatmap()
			self.labels = label(output_heatmap)
		return self.labels















