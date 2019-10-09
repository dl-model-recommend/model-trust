'''
Generating multiple class synthetic dataset with balanced samples
'''

import sys
sys.path.append("..")

from make_dataset import dataset_params
from utils import util_functions as utils
import os, random, cv2, math, csv, pandas as pd, numpy as np

def make_npz_file(data_type):
	"""Save the generated images as compressd numpy (.npz) file"""

	data_folder = data_type + "_images"
	label_file = os.path.join(dataset_params.data_path, data_type + "_lables.csv")
	output_file = os.path.join(dataset_params.data_path, "synthetic_" + data_type + "_data")
	line_reader = csv.DictReader(open(label_file,"r"))

	data = []
	labels = []
	data_points = 0
	for row in line_reader:
		image_name = os.path.join(dataset_params.data_path,data_folder,row["figNum"] + ".png")
		image_data = cv2.imread(image_name, cv2.IMREAD_COLOR)
		image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
		image_label = [int(dataset_params.shapes[row["shape"]]), int(dataset_params.colors[row["color"]]), int(dataset_params.sizes[row["size"]]), int(row["quadrant"]), int(dataset_params.backgrounds[row["background"]]) ]
		data.append(image_data)
		labels.append(image_label)
		data_points += 1

	# Converting list to data to np array
	data = np.asarray(data)
	labels = np.asarray(labels)

	# Printing log information
	print(data_type, "statistics being saved: ")
	print(data_type, "data shape", data.shape)
	print(data_type, "label shape", labels.shape)

	# saveing the file as npz file
	np.savez_compressed(output_file, data=data, lables=labels)

def polygon(center, sides, radius=1, rotation=0, translation=None):
	"""This is the function that creates the polygon points for makePolygon function"""

	one_segment = math.pi * 2 / sides
	points = [
		(int(round(center[0] + math.sin(one_segment * i + rotation) * radius, 0)),
		 int(round(center[1] + math.cos(one_segment * i + rotation) * radius, 0)))
		for i in range(sides)]
	if translation:
		points = [[sum(pair) for pair in zip(point, translation)]
				  for point in points]
	return points

def makePolygon(center, sides, radius, background, colorValue, colorsRGB):
	"""This is the function that creates the cv2 polygon object"""

	points = polygon(center, sides, radius)
	pointsList = [list(a) for a in points]
	p1 = np.array(pointsList)
	img = np.zeros((256, 256, 3), dtype='int32')
	if(background == "white"):
		img.fill(255)
	elif(background == "random"):
		r = random.randint(200,245)
		b = random.randint(200,245)
		g = random.randint(200,245)
		img = np.full(img.shape, (r,b,g), dtype=np.uint8)
	cv2.fillPoly(img, pts =[p1], color = colorsRGB[colorValue])
	return img

def makeCircle(presentColor, radius, center, background, colorsRGB):
	"""This is the function that creates the cv2 circle object"""

	img = np.ones((dataset_params.image_size_x,dataset_params.image_size_y,3), np.uint8)
	if(background == "white"):
		img.fill(255)
	elif(background == "random"):
		r = random.randint(200,245)
		b = random.randint(200,245)
		g = random.randint(200,245)
		img = np.full(img.shape, (r,b,g), dtype=np.uint8)
	cv2.circle(img, (center[0], center[1]), radius, colorsRGB[presentColor], -1)
	return img

def makeDataset(numberOfTrials, data_type):
	"""Make train or test dataset based on the number of samples given"""

	data_folder = data_type + "_images"
	label_file = os.path.join(dataset_params.data_path, data_type + "_lables.csv")

	utils.create_directory(dataset_params.data_path)
	utils.create_directory(os.path.join(dataset_params.data_path, data_folder))

	allowedRadius = utils.defineShapePerimeter()
	colorsRGB = utils.defineColorValues()
	shapeDict = utils.defineShapeSides()
	padding = dataset_params.padding

	num = 0
	output_images = [["figNum", "shape", "color", "size", "background", "quadrant", "radius"]]
	for c in dataset_params.colors: # for all 7 foreground colors 
		for q in dataset_params.quadrants: # for all 4 quadratns 
			for s in dataset_params.shapes: # for all 5 shapes
				for k in dataset_params.sizes: # for all 3 sizes
					for b in dataset_params.backgrounds: # for all 3 background colors
						for i in range(numberOfTrials):
							fileName = os.path.join(dataset_params.data_path, data_folder, str(num) + ".png")
							presentQuadrant = dataset_params.quadrants[q]
							radius = random.randint(allowedRadius[s][k][0],allowedRadius[s][k][1])

							if(presentQuadrant == 3):
								xMin = 128 + padding
								xMax = 255 - radius
								yMin = 128 + padding
								yMax = 255 - radius

							elif(presentQuadrant == 2):
								xMin = 0 + radius
								xMax = 128 - padding
								yMin = 128 + padding
								yMax = 255 - radius

							elif(presentQuadrant == 1):
								xMin = 0 + radius
								xMax = 128 - padding
								yMin = 0 + radius
								yMax = 128 - padding

							else:
								xMin = 128 + padding
								xMax = 255 - radius
								yMin = 0 + radius
								yMax = 128 - padding

							xCenter = random.randint(xMin, xMax)
							yCenter = random.randint(yMin, yMax)
							center = [xCenter, yCenter]

							if(s == "circle"):
								output_images.append([num, "circle", c, k, b, presentQuadrant, radius])
								img = makeCircle(c, radius, center, b, colorsRGB)
								img = img[:,:,::-1]
								cv2.imwrite(fileName, img)
							else:
								n = shapeDict[s]
								img = makePolygon(center, n, radius, b, c, colorsRGB)
								img = img[:,:,::-1]
								cv2.imwrite(fileName, img)
								output_images.append([num, s, c, k, b, presentQuadrant, radius])
							num += 1
	
	print("Number of image generated", num)

	print("Saving " + data_type +  " data meta information to CSV ......")
	df = pd.DataFrame(output_images[1:], columns=output_images[0])
	df.to_csv(label_file, index=False)
	print("Saved " + data_type +  " data meta information: " + data_folder)
	

	print("Saving " + data_type +  " images data to npz(numpy) compressed file ......")
	make_npz_file(data_type)
	print("Saved " + data_type +  " images data to npz(numpy) compressed file!")
	
	return None

def make_dataset():
	"""This is the main function that creates the dataset"""

	numberOfTrials = dataset_params.num_of_samples
	numberOfTrials_train = int(numberOfTrials*0.8)
	numberOfTrials_test = int(numberOfTrials*0.2)

	print("==================================================")
	print("1. Generating Train images ......")
	print("\nTrain image per variation", numberOfTrials_train)
	makeDataset(numberOfTrials_train, "train")

	print("==================================================")
	print("2. Generating Test images ......")
	print("\nTest image per variation", numberOfTrials_test)
	makeDataset(numberOfTrials_test, "test")

	print("==================================================")
	print("Done!!!")

if __name__ == '__main__':
	make_dataset()