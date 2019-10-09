########################################
# Dataset Params
#######################################

# Minimum number of image samples to generate per variation in the dataset
num_of_samples = 5 # Keep 5 as the min number. Anything above 5 works

# Image size of each image
image_size_x = 256
image_size_y = 256

# Image object geometrical placement parameters
padding = 25 

# Root data path where the simulated data will be generated and saved
data_path = "./data/"

# Define all the tasks and the number of classes in each task
# Task #1 - Shape Classification (5 classes)
shapes = {
	"circle": 0,
	"quadrilateral": 1,
	"triangle": 2,
	"pentagon": 3,
	"hexagon": 4
}

# Task #2 - Shape Color Classification (7 classes)
colors = {
	"violet": 0,
	"indigo": 1,
	"blue": 2,
	"green": 3,
	"yellow": 4,
	"orange": 5,
	"red": 6
}

# Task #3 - Size Classification (3 classes)
sizes = {
	"small": 0,
	"medium": 1,
	"large": 2
}

# Task #4 - Background Color Classification (3 classes)
backgrounds = {
	"white": 0,
	"black": 1,
	"random": 2
}

# Task #5 - Quadrant Location Classification (4 classes)
quadrants = {
	"upper_right": 0,
	"upper_left": 1,
	"bottom_left": 2,
    "bottom_right": 3
}

