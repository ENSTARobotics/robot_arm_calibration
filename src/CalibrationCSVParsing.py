#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
import csv
from collections import OrderedDict

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

### USEFULL FUNCTION

def csvRowToArray(csvRow):
    arrayRow = []
    for csvItem in csvRow:
        csvItem.replace(',','.')
        arrayItem = float(csvItem)
        arrayRow.append(arrayItem)
    return(np.array(arrayRow))

### CSV PARSING

def calibrationCSVParsing(itemsNames,calibrationCSVPath):

    Items = OrderedDict.fromkeys(itemsNames,np.empty((0,0)))

    with open(calibrationCSVPath, "r") as csvfile:
        rowCounter = 0
        rowReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rowReader:
            if(row[0]=="---"):
                rowCounter = 0
            else:
                currentRow = csvRowToArray(row)
                if(np.shape(Items[itemsNames[rowCounter]]) == (0,0)):
                    Items[itemsNames[rowCounter]] = np.array([currentRow])
                else:
                    Items[itemsNames[rowCounter]] = np.append(Items[itemsNames[rowCounter]],np.array([currentRow]),axis=0)
                rowCounter+=1

    return(Items)
