#!/usr/bin/env python

import sys
import cv2

sys.path.append("../build")

from pydbow2 import PyDBoW2


ims = []
dbow2 = PyDBoW2('../Vocabulary/ORBvoc.txt')

for i in range(4):
	ims.append(cv2.cvtColor(cv2.imread( "images/image" + str(i) + ".png" ), cv2.COLOR_BGR2GRAY))
        dbow2.addToDB(ims[-1])

k=0
for im in ims:
	ind, score = dbow2.getClosestMatch(im)
	print "Image ", k, ", query ind =", ind, ", score =", score 
	k += 1

