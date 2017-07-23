""" This file takes in a csv file having Soundex Code, HAC classifier ID, Row ID Author Name and disambiguates every author name"""
# 10th Soundex Cluster Chosen by default for example. Change it if required in line number 106.

# Importing necessary packages

import csv
import os
import sys
import string
from collections import defaultdict
import time
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np

# Function to get LDA
def getlda(doc_a):

	tokenizer = RegexpTokenizer(r'\w+')

	# create English stop words list
	en_stop = get_stop_words('en')

	# Create p_stemmer of class PorterStemmer
	p_stemmer = PorterStemmer()
    
	# create sample documents
	#doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."

	# compile sample documents into a list
	doc_set = [doc_a]

	# list for tokenized documents in loop
	texts = []

	# loop through document list
	for i in doc_set:

		# clean and tokenize document string
    		raw = i.lower()
    		tokens = tokenizer.tokenize(raw)

    		# remove stop words from tokens
    		stopped_tokens = [i for i in tokens if not i in en_stop]
    
    		# stem tokens
    		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    		# add tokens to list
    		texts.append(stemmed_tokens)

	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)
    
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]

	# generate LDA model
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word = dictionary, passes=20)


	x =ldamodel.print_topics(num_topics=1, num_words=10)

	words =x[0][1].split("+")

	for loop5 in xrange (0,len(words)):
		begin = words[loop5].find("\"")
		end = words[loop5].rfind("\"")
		words[loop5] =  words[loop5][begin+1:end]


	return words


# Open Files to read and write.
input = open("Level1-2.csv", 'rb')
file = open("Results.txt","w") 
d = defaultdict(list)

# List to store the all soundex codes in the file
soundexCodeList = []

rowCount = 0

print "Fetching all Soundex hash codes"
time.sleep(1)
for row in csv.reader(input):

	if rowCount == 0:
		rowCount = 1
		continue 

	print "Processing Row:",rowCount

	# Check for a new input
	if row[0] not in soundexCodeList:
		soundexCodeList.append(row[0])

	# Counter for next row
	rowCount = rowCount+1

# Printing total number of HAC clusters
print "Number of Soundex Clusters in the file:",len(soundexCodeList)

# Take any cluster, and then we try to normalize all authors in the cluster
soundexCode = soundexCodeList[10]
print "We will use soundex code:",soundexCode
print "Fetching number of HAC clusters"

input.close()
input = open("Level1-2.csv", 'rb')

# Fetching number of HAC clsuters in this Level-1 Soundex Cluster
rowCount = 0
noOfClusters = 0

for row in csv.reader(input):

	if rowCount == 0:
		rowCount = 1
		continue 

	print "Processing Row:",rowCount

	# Check for a new input
	if row[0] == soundexCode:
		if int(row[1]) > noOfClusters:
			noOfClusters = int(row[1])

	# Counter for next row
	rowCount = rowCount+1

print "Number of HAC Clusters in ",soundexCode, "is ",noOfClusters

input.close()

#Storing the information corresponding to each cluster in dictionary
for i in xrange(1,1+noOfClusters):

	input = open("Level1-2.csv", 'rb')
	rowCount = 0
	for row in csv.reader(input):

		if rowCount == 0:
			rowCount = 1
			continue

		if row[0] == soundexCode and row[1]==str(i):
			d[i].append(row[2]+"+"+row[3])

		rowCount = rowCount + 1
	input.close()

for i in xrange(1,1+noOfClusters):
	
	noOfItemsInCluster = len(d[i])
	for loop7 in xrange(0,noOfItemsInCluster):
		if loop7 == 0:
			print "\n\nNames in cluster number ",i,"is/are:"
		xy = d[i][loop7].split("+")
		print "[ID:",xy[0],"] ",xy[1]," "
	#print 
	print "Number of items in cluster number ",i,"is",noOfItemsInCluster
	# List to store cluster ids
	id = []
	# List to store name of authors
	authorNames = []
	# Matrix to store ambiguity/disambiguity status
	w, h = noOfItemsInCluster, noOfItemsInCluster
	Matrix = [[0 for x in range(w)] for y in range(h)] 
	for loop1 in range(0,noOfItemsInCluster):
		for loop2 in range(0,noOfItemsInCluster):
			if  loop1!=loop2:
				Matrix[loop1][loop2] = 0
			else:
				Matrix[loop1][loop2] = 1

	for j in xrange(0,noOfItemsInCluster):
		infoID, infoName = (d[i][j]).split("+")
		id.append(infoID)
		authorNames.append(infoName)


	# No disambiguity incase there is only one element
	if len(id) == 1:
		
		Matrix[0][0] = 1

	else:

		firstName = []
		lastName = []
		for k in xrange(0,noOfItemsInCluster):
			index = authorNames[k].rfind(" ")
			if index!=-1:
				first = (authorNames[k])[0:index]
				last = (authorNames[k])[index+1:]
				firstName.append(first)
				lastName.append(last)
			else:
				firstName.append("")
				lastName.append(authorNames[k])

		# Check for exact last name, if not same we can say they are different
		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if lastName[loop1]!=lastName[loop2]:
					Matrix[loop1][loop2] = -1
					Matrix[loop2][loop1] = -1

		# Check if the first letter of first name matches or not
		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if Matrix[loop1][loop2] == 0 and Matrix[loop2][loop1] == 0:
					if (firstName[loop1])[0] != (firstName[loop2])[0] and (firstName[loop1])[0] != "" and (firstName[loop2])[0] != "":
						Matrix[loop1][loop2] = -1
						Matrix[loop2][loop1] = -1

		# If there are no dots and spaces in firstname is same, then first names should match.
		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if Matrix[loop1][loop2] == 0 and Matrix[loop2][loop1] == 0:
					index1 = firstName[loop1].find(".") 
					index2 = firstName[loop2].find(".")
					if index1 == -1 and index2 == -1:
						noOfSpaces1 = firstName[loop1].count(' ')
						noOfSpaces2 = firstName[loop2].count(' ')
						if noOfSpaces1 == noOfSpaces2:
							if firstName[loop1]!= firstName[loop2]:
								Matrix[loop1][loop2] = -1
								Matrix[loop2][loop1] = -1



		# After the normal heuristics, we take in 4 fields from Master File
		# We take in coauthors,publication date, abstract and affiliation.
		# First we check if the author entry has any coauthor same. If yes there is a chance that the both entries are of the same person
		# Now we can check for affiliation and use LDA on that, if there is a match, we may conclude both the entries are same
		# We check for the abstract name, using LDA we try to match. If the topic names match we may be the same person
		# Then we check for year of publication, if the publication years are within a difference of 30 years. We may say all are same
		# If none of the fields match, we ask the user for hand clustering
		# Though it is a very weak framework

		# First up we check for co authors,if there is a match, we say both names are same

		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if Matrix[loop1][loop2] >= 0 and Matrix[loop2][loop1] >=0 and loop1!=loop2 and loop2>loop1:
					id1 = id[loop1] 
					id2 = id[loop2]
					
					# Now we search the id in master file to get data
					input = open("NamesNormalized.csv", 'rb')
					rowCount = 0
					flag = 0
					for row in csv.reader(input):
						if flag == 2:
							break

						if rowCount==0:
							rowCount=1
							continue

						if rowCount == int(id1):
							CoauthorList1 = row[1]
							flag = flag + 1
							

						if rowCount == int(id2):
							CoauthorList2 = row[1]
							flag =flag + 1
							
						rowCount = rowCount +1 

					CoAuthorList1Array = CoauthorList1.split(":")
					CoAuthorList2Array = CoauthorList2.split(":")
					
					flagCoauthor = 0
					for loop3 in xrange (1, len(CoAuthorList1Array)):
						for loop4 in xrange (1,len(CoAuthorList2Array)):
							if CoAuthorList1Array[loop3] == CoAuthorList2Array[loop4]:
								flagCoauthor = flagCoauthor + 1
								#print "Coauthor Match"," ","[ID:",id[loop1],"]","[ID:",id[loop2],"]:",CoAuthorList1Array[loop3]
					if flagCoauthor > 0:
						Matrix[loop2][loop1] = Matrix[loop2][loop1] + 0.3
						Matrix[loop1][loop2] = Matrix[loop1][loop2] + 0.3

		input.close()
		
		# Check for  description of author

		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if Matrix[loop1][loop2]>= 0 and Matrix[loop2][loop1] >=0 and loop1!=loop2 and loop2>loop1:
					id1 = id[loop1] 
					id2 = id[loop2]
					
					# Now we search the abstract in master file to get data
					input = open("NamesNormalized.csv", 'rb')
					rowCount = 0
					flag = 0
					for row in csv.reader(input):
						if flag == 2:
							break

						if rowCount==0:
							rowCount=1
							continue

						if rowCount == int(id1):
							details1 = row[4]
							flag = flag + 1
							

						if rowCount == int(id2):
							details2 = row[4]
							flag =flag + 1
							
						rowCount = rowCount +1

					details1 = details1.replace("Author affiliation","")
					details1 = details1.replace("||","")


					details2 = details2.replace("Author affiliation","")
					details2 = details2.replace("||","")

					# Removing Non Ascii characters
					details1 = ''.join(i for i in details1 if ord(i)<128)
					details2 = ''.join(i for i in details2 if ord(i)<128)

					if details1 != "":
						ldaWords1 = getlda(details1)
					else:
						ldaWords1 = []

					if details2 != "":
						ldaWords2 = getlda(details2)
					else:
						ldaWords2 = []
					flagLda = 0
					for loop3 in xrange (0, len(ldaWords1)):
						for loop4 in xrange (0,len(ldaWords2)):
							if ldaWords1[loop3] == ldaWords2[loop4]:
								flagLda = flagLda + 1
								#print "Descrption Keyword Match"," ","[ID:",id[loop1],"]","[ID:",id[loop2],"]:",ldaWords1[loop3]
					if flagLda > 0:
						#print "Number of matches",flagLda
						Matrix[loop1][loop2] = Matrix[loop1][loop2] + 0.3
						Matrix[loop2][loop1] = Matrix[loop2][loop1] + 0.3

		input.close()
						
		# Check for abstract

		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if Matrix[loop1][loop2] == 0 and Matrix[loop2][loop1] ==0 and loop1!=loop2 and loop2>loop1:
					id1 = id[loop1] 
					id2 = id[loop2]
					
					# Now we search the abstract in master file to get data
					input = open("NamesNormalized.csv", 'rb')
					rowCount = 0
					flag = 0
					for row in csv.reader(input):
						if flag == 2:
							break

						if rowCount==0:
							rowCount=1
							continue

						if rowCount == int(id1):
							abstract1 = row[3]
							flag = flag + 1
							

						if rowCount == int(id2):
							abstract2 = row[3]
							flag =flag + 1
							
						rowCount = rowCount +1

					# Removing Non Ascii characters
					abstract1 = ''.join(i for i in abstract1 if ord(i)<128)
					abstract2 = ''.join(i for i in abstract2 if ord(i)<128)

					if abstract1!="":
						ldaWords1 = getlda(abstract1)
					else:
						ldaWords1 = []

					if abstract2!="":
						ldaWords2 = getlda(abstract2)
					else:
						ldaWords2 = []
					flaglda = 0
					for loop3 in xrange (0, len(ldaWords1)):
						for loop4 in xrange (0,len(ldaWords2)):
							if ldaWords1[loop3] == ldaWords2[loop4]:
								flaglda = flaglda + 1
								#print "Abstract Keyword Match"," ","[ID:",id[loop1],"]","[ID:",id[loop2],"]:",ldaWords1[loop3]
					if flaglda > 0:
						#print "Number of matches",flaglda
						Matrix[loop1][loop2] = Matrix[loop1][loop2] + 0.3
						Matrix[loop2][loop1] = Matrix[loop2][loop1] + 0.3

		input.close()
		


		# Now we check for publication date

		for loop1 in xrange(0,noOfItemsInCluster):
			for loop2 in xrange(0,noOfItemsInCluster):
				if Matrix[loop1][loop2] >= 0 and Matrix[loop2][loop1] >=0 and loop1!=loop2 and loop2>loop1:
					id1 = id[loop1] 
					id2 = id[loop2]
					
					# Now we search the id in master file to get data
					input = open("NamesNormalized.csv", 'rb')
					rowCount = 0
					flag = 0
					for row in csv.reader(input):
						if flag == 2:
							break

						if rowCount==0:
							rowCount=1
							continue

						if rowCount == int(id1):
							date1 = int(row[2])
							flag = flag + 1
							

						if rowCount == int(id2):
							date2 = int(row[2])
							flag =flag + 1
							
						rowCount = rowCount +1 

					diff = date1 - date2
					if diff < 0:
						diff = -diff

					# can be changed depending on cases
					if diff < 20:
						Matrix[loop1][loop2] = Matrix[loop1][loop2] + 0.1
						Matrix[loop2][loop1] = Matrix[loop2][loop1] + 0.1
		input.close()

	MatrixCopy = Matrix
	#print "Normalized names"
	"""
	for loop1 in xrange(0,noOfItemsInCluster):
		if noOfItemsInCluster == 1:
			print "Only[ID:",id[loop1],"]",authorNames[loop1]," is present"
			break
		else:
			flag = 0
			print "[ID:",id[loop1],"]",authorNames[loop1],
			for loop2 in xrange(0,noOfItemsInCluster):
				if loop1!=loop2:
					if Matrix[loop1][loop2] > 0 :
						if flag == 0:
							print "is compatible with: ",
						flag = 1
						print "[ID:",id[loop2],"]",authorNames[loop2]," [",Matrix[loop1][loop2],"] ",
			
			if flag == 0 :
				print " No possible grouping found",
			print ""
	"""
	print "Normalized Names"
	d1=[]
	for loop1 in xrange(0,noOfItemsInCluster):
		if noOfItemsInCluster == 1:
			print "[ID:",id[loop1],"]",authorNames[loop1]
			break
		else:
			flag = 0
			flagVal = 0
			flagVal2 = 0 
			for loop2 in xrange(0,noOfItemsInCluster):
				if MatrixCopy[loop1][loop2] > 0 and MatrixCopy[loop1][loop2] <= 1.0 and loop1!=loop2:
					if flag == 0:
						if loop1 not in d1:
							str1 = "[ID:"+id[loop1]+"] "+authorNames[loop1]+" "
							#print str1,
							file.write(str1)
							d1.append(loop1)
							flag = 1
					if loop2 not in d1:
						str2 = "[ID:"+id[loop2]+"] "+authorNames[loop2]+" "#" [",Matrix[loop1][loop2],"] ",
						#print str2,
						file.write(str2)
						flagVal = 1
						MatrixCopy[loop2][loop1] = 5.0
						d1.append(loop2)
			if flagVal ==1:
				file.write("\n")
			if flagVal == 0:
				for loop3 in  xrange(0,noOfItemsInCluster):
					if MatrixCopy[loop1][loop3] == 5.0:
						flagVal2 = 1
				if flagVal2 == 0 and loop1 not in d1:
					str3 = "[ID:"+id[loop1]+"] "+authorNames[loop1]
					#print str3
					file.write(str3)
					file.write("\n")
					d1.append(loop1)