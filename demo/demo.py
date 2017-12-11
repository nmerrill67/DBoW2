#!/usr/bin/env python

import sys
import cv2

sys.path.append("../build")

from dbow2 import PyDBoW2


ims = []
dbow2 = PyDBoW2()

for i in range(4):
	ims.append(cv2.cvtColor(cv2.imread( "images/image" + str(i) + ".png" ), cv2.COLOR_BGR2GRAY))
	dbow2.addVoc(ims[i])

print "Added images to vocab"

dbow2.createVocAndDB()	

k=0
for im in ims:
	ind, score = dbow2.getClosestMatch(im)
	print "Image ", k, ", query ind =", ind, ", score =", score 
	k += 1

