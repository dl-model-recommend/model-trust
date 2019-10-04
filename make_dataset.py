'''
Create synthetic dataset for model trust experiments

'''

import argparse
import csv
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('-n', '--num_of_samples', type=int, help='Number of images to generate per variation', required=True)
args = parser.parse_args()

data_path = "./data/"
if not os.path.exists(data_path):
	os.makedirs(data_path)

colorsRGB = {
	"violet": (148, 0, 211),
	"indigo": (75, 0, 130),
	"blue": (0, 0, 255),
	"green": (0, 255, 0),
	"yellow": (255, 255, 0),
	"orange": (255, 127, 0),
	"red": (255, 0, 0),
}
colors = list(colorsRGB.keys())

# In[4]:

# In[5]:


def polygon(center, sides, radius=1, rotation=0, translation=None):
	one_segment = math.pi * 2 / sides

	points = [
		(int(round(center[0] + math.sin(one_segment * i + rotation) * radius, 0)),
		 int(round(center[1] + math.cos(one_segment * i + rotation) * radius, 0)))
		for i in range(sides)]

	if translation:
		points = [[sum(pair) for pair in zip(point, translation)]
				  for point in points]

	return points


# In[7]:


def makePolygon(center, sides, radius, background, colorValue):
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
	cv2.fillPoly(img, pts =[p1], color=colorsRGB[colorValue])
	return img


# In[8]:


shape = {
	"circle": 0,
	"quadrilateral": 1,
	"triangle": 2,
	"pentagon": 3,
	"hexagon": 4
}

colors = {
	"violet": 0,
	"indigo": 1,
	"blue": 2,
	"green": 3,
	"yellow": 4,
	"orange": 5,
	"red": 6
}

size = {
	"small": 0,
	"medium": 1,
	"large": 2
}

backgrounds = {
	"white": 0,
	"black": 1,
	"random": 2
}

def makeCircle(presentColor, radius, center, background):
	img = np.ones((256,256,3), np.uint8)
	if(background == "white"):
		img.fill(255)
	elif(background == "random"):
		r = random.randint(200,245)
		b = random.randint(200,245)
		g = random.randint(200,245)
		img = np.full(img.shape, (r,b,g), dtype=np.uint8)

	cv2.circle(img,(center[0], center[1]), radius, colorsRGB[presentColor], -1)
	return img

def make_npz_file(data_type):

	data_folder = data_type + "_images"
	lable_file = os.path.join(data_path, data_type + "_lables.csv")
	output_file = os.path.join(data_path, "synthetic_" + data_type + "_data")

	line_reader = csv.DictReader(open(lable_file,"r"))

	data = []
	lables = []
	data_points = 0

	for row in line_reader:

		image_name = os.path.join(data_path,data_folder,row["figNum"] + ".png")
		image_data = cv2.imread(image_name, cv2.IMREAD_COLOR)
		image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

		image_lable = [ int(shape[row["shape"]]), int(colors[row["color"]]), int(size[row["size"]]), int(row["quadrant"]), int(backgrounds[row["background"]]) ]

		data.append(image_data)
		lables.append(image_lable)
		data_points += 1

	# Converting list to data to np array
	data = np.asarray(data)
	lables = np.asarray(lables)

	# Printing log information
	print("Num of data processed", data_points, "in dataset", data_type)
	print(data_type, "data shape", data.shape)
	print(data_type, "label shape", lables.shape)

	# saveing the file as npz file
	np.savez_compressed(output_file, data=data, lables=lables)
	print(data_type, "compressed file store at path", output_file)

def makeDataset(numberOfTrials, data_type):

	data_folder = data_type + "_images"
	lable_file = os.path.join(data_path, data_type + "_lables.csv")
	try:
		os.mkdir(os.path.join(data_path, data_folder))
	except:
		print(data_type, "data folder exits")

	backgrounds = ["white", "black", "random"]
	quadrants = [0,1,2,3]
	# quadrant = 1
	sizesDict = {1 : "small", 2 : "medium", 3:"large"}
	size = ["small", "medium", "large"]
	# size = ["small"]
	answers = [["figNum", "shape", "color", "size", "background", "quadrant", "radius"]]
	shapes = ["circle", "quadrilateral", "triangle", "pentagon", "hexagon"]
	shapeDict = {"quadrilateral" : 4, "triangle" : 3, "pentagon" : 5, "hexagon" : 6}
	tic = time.time()
	allowedRadius = {
					"circle" : {
						"small" : [16,25],
						"medium" : [32,40],
						"large" : [45,58]
					},
					"quadrilateral" : {
						"small" : [16,32],
						"medium" : [40,48],
						"large" : [56,72]
					},
					"triangle" : {
						"small" : [20,38],
						"medium" : [50,60],
						"large" : [70,88]
					},
					"pentagon" : {
						"small" : [14,28],
						"medium" : [36,44],
						"large" : [52,68]
					},
					"hexagon" : {
						"small" : [14,28],
						"medium" : [35,44],
						"large" : [49,64]
					}

					}
	sizeImages = [256,256]
	xRange = [0,0]
	yRange = [0,0]
	center = [0,0]
	radius = 0
	num = 0
	# quadrantRange = {1 : }
	# b = "white"
	padding = 25
	for c in colors:#for all 7 colors #7
		for q in quadrants:#4
			for s in shapes:#5
				for k in size:#3
					for b in backgrounds:
						for i in range(numberOfTrials):
		#                     try:
							fileName = os.path.join(data_path, data_folder, str(num) + ".png")
			#                 presentQuadrant = (quadrant % 4) + 1
							presentQuadrant = q
		#                     print(allowedRadius[s][k][0])
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
								answers.append([num, "circle", c, k, b, presentQuadrant, radius])
								img = makeCircle(c, radius, center, b)
								img = img[:,:,::-1]
								cv2.imwrite(fileName, img)
							else:
								n = shapeDict[s]
								img = makePolygon(center, n, radius, b, c)
								img = img[:,:,::-1]
								cv2.imwrite(fileName, img)
								answers.append([num, s, c, k, b, presentQuadrant, radius])
			#                 quadrant += 1
							num += 1
		#                     except Exception as e:
		#                         print(num, repr(e))
		# print(time.time() - tic, "completed color : ", c)


	# print("Number of lines in CSV including heards", len(answers))#this is with header
	print("Number of image generated", num)

	print("Saving", data_type, "raw image")
	df = pd.DataFrame(answers[1:], columns=answers[0])
	df.to_csv(lable_file, index=False)
	print(data_type, "raw image saved at path", data_folder)

	print("\nPreparing", data_type, "npz(numpy) compressed file")
	make_npz_file(data_type)
	print(data_type, "data preparation finished")

def main():
	numberOfTrials = args.num_of_samples
	numberOfTrials_train = int(numberOfTrials*0.8)
	numberOfTrials_test = int(numberOfTrials*0.2)

	print("Total image per variation", numberOfTrials)

	print("\nTrain image per variation", numberOfTrials_train)
	makeDataset(numberOfTrials_train, "train")

	print("\nTest image per variation", numberOfTrials_test)
	makeDataset(numberOfTrials_test, "test")

if __name__ == '__main__':
	main()
