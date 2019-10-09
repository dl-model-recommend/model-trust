'''
A collection of differen utility functions
'''
import os

def create_directory(dir_path):
    """Create directory if it does not exist"""

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return None

def defineShapeSides():
    """Define the number of sides for each shape"""

    shapeDict = {
        "quadrilateral" : 4, 
        "triangle" : 3, 
        "pentagon" : 5, 
        "hexagon" : 6
    }
    return shapeDict

def defineColorValues():
	"""Define the RGB color values for different colors"""

	colorsRGB = {
		"violet": (148, 0, 211),
		"indigo": (75, 0, 130),
		"blue": (0, 0, 255),
		"green": (0, 255, 0),
		"yellow": (255, 255, 0),
		"orange": (255, 127, 0),
		"red": (255, 0, 0),
	}
	return colorsRGB

def defineShapePerimeter():
	"""Define the perimeter and radius of each shape for different sizes"""

	allowedRadius = {
		"circle" : {
			"small" : [16,25], "medium" : [32,40], "large" : [45,58]
		},
		"quadrilateral" : {
			"small" : [16,32], "medium" : [40,48], "large" : [56,72]
		},
		"triangle" : {
			"small" : [20,38], "medium" : [50,60], "large" : [70,88]
		},
		"pentagon" : {
			"small" : [14,28], "medium" : [36,44], "large" : [52,68]
		},
		"hexagon" : {
			"small" : [14,28], "medium" : [35,44], "large" : [49,64]
		}
	}
	return allowedRadius