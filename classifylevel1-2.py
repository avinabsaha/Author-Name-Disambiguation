""" This code takes in csv file with format Author Name, Co-author Names(if available) and other details
 and and does two levels of clustering Soundex and HAC"""


# Importing necessary packages

import phonetics as ph
import csv
import string
import numpy as np 
import time
from collections import defaultdict
from jellyfish import jaro_distance
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import time
import sys
import os

# Some fields have large values, so in order to maximize filed size limitm this command is used.
csv.field_size_limit(sys.maxsize)


# Function to Calculate Soundex hash code
def getSoundex(lastN):
	lastN = string.capwords(lastN)
	if (lastN.isalpha() == True):
		hashcode = ph.soundex(lastN)
		hashcode = hashcode + '0000'
		hashcode = hashcode[0:4]
		return hashcode
	return "None"

# Calculate time	
start_time = time.time()

# Open Files to read and write.
input = open("NamesNormalized.csv", 'rb')
output = open("Level1-2.csv",'wb')
writer = csv.writer(output)
d = defaultdict(list)

# Count for index number in the row
rowCount = 0

# Level 1 Soundex Clustering
print "First level Soundex hashing"

# Sleep for better illustration
time.sleep(2)

for row in csv.reader(input):

	if rowCount == 0:
		rowCount = rowCount + 1
		continue

	print "Processing entry number",rowCount
	


	# extracting information information
	authorName = row[0]
	#coAuthorName = row[1]
	#copyRightDate = row[2]
	#abstractName = row[3]
	#description = row[4]

	# Sleep for better illustration
	#time.sleep(0.01)
    
	# Split the words into different parts
	partsOfName = authorName.split()

	# Extracting last name
	lastName = partsOfName[len(partsOfName)-1]
	lastName = lastName.strip()

	# Finding hash from getSoundex function
	soundexCode = getSoundex(lastName)

	# All info clustered into one separated by "+" (Format: Row index + AuthorName #+ CoAuthor Name+ Copyright Data + Abstract name + Description)
	infoClassifierLevel1 = str(rowCount)+"+"+authorName#+"+"+coAuthorName+"+"+copyRightDate+"+"+abstractName+"+"+description
	
	# Adding to dictionary
	d[soundexCode].append(infoClassifierLevel1)

	# Index count
	rowCount = rowCount + 1

numberOfSoundexClusters = len(d)

# Number of first level clusters
print "Number of First Level Soundex Clusters are: ",numberOfSoundexClusters
print "Second Level HAC clustering"
# Sleep for better illustration
time.sleep(3)

# Level 2 HAC Clustering. Take input each Soundex Cluster and forms clusters inside it.

for i in xrange (0,numberOfSoundexClusters):
	print "Processing Hash index:",i+1,
	print "Processing Hash value:",d.keys()[i],

	numberOfDataPoints = len(d.values()[i])
	print "Number of datapoints inside the hash", numberOfDataPoints

	dataPoints = d.values()[i]

	if numberOfDataPoints==1:
		(d.values()[i])[0] = str(1)+"+"+ (d.values()[i])[0]
		continue

	# HAC clustering inside each soundex cluster.
	surNames = []
	for j in xrange(0,numberOfDataPoints):
		# Extact name
		tokens = dataPoints[j].split("+")
		name = tokens[1]
		parts = name.split()
		lastPart = parts[len(parts)-1]
		surNames.append(lastPart)

	# Convert List to array
	surNamesArray = np.asarray(surNames)

	# Distance calculations using jaro winkler technique
	def dist(coord):
		i, j = coord
		one = unicode(surNamesArray[i], "utf-8")
		two = unicode(surNamesArray[j], "utf-8")
		return 1 - jaro_distance(one,two)

	# Now we use clustering
	upperTriangle = np.triu_indices(numberOfDataPoints, 1)
	distance = np.apply_along_axis(dist, 0, upperTriangle)
	Z = scipy.cluster.hierarchy.linkage(distance,method='single')

	K = scipy.cluster.hierarchy.fcluster(Z,0.5)
	max = np.amax(K)
	for loop1 in xrange(1,1+max):
		#print loop1,":",
		for loop2 in xrange(0,len(K)):
			if K[loop2]==loop1:
				#print surNames[loop2],
				# Data in form of Key:HashIndex Values: 2ndlevelClassificationValue+RowCount #+Authorname+Abstractname+Coauthor
				(d.values()[i])[loop2] = str(loop1)+"+"+(d.values()[i])[loop2]
		#print

print "Second Level Clustering Done"
print("--- Time Required %s seconds ---" % (time.time() - start_time))


print("Printing to Dump File")



rowList = []
rowList.append("Soundex Code")
rowList.append("Cluster ID")
rowList.append("Row ID")
rowList.append("Author Name")
writer.writerow(rowList)

# Classification print function
for i in xrange(0,numberOfSoundexClusters):
	#print d.keys()[i]
	for j in xrange(0,len(d.values()[i])):
		
		rowList = []
		info =  (d.values()[i][j]).split("+")	
		rowList.append(d.keys()[i])
		rowList.append(info[0])
		rowList.append(info[1])
		rowList.append(info[2])
		writer.writerow(rowList)
		
print("Printing to Dump File Level1-2.csv, Done")
