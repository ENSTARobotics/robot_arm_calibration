#!/usr/bin/python3

#System packages
import csv
from collections import OrderedDict

#Utility packages
import numpy as np

### Tool function

def CSVRowToArray(CSVRow):
    """! Converts a CSV extracted row into a numpy compatible array.

    @param CSVRow The CSV row.

    @return  The corresponding numpy array.
    """
    arrayRow = []
    for csvItem in CSVRow:
        csvItem.replace(',','.')
        arrayItem = float(csvItem)
        arrayRow.append(arrayItem)
    return(np.array(arrayRow))

### CSV parsing function

def calibrationCSVParsing(itemsNames,calibrationCSVPath):
    """! Stores the contents of a calibration measurements CSV into a python ordered dict.

    @param itemsNames           The names of the dict keys corresponding to the CSV data. Usually : ["pose1",..."poseN","jointsPositions"]
    @param calibrationCSVPath   The path the the calibration CSV file.

    @return  The corresponding python ordered dict.
    """
    Items = OrderedDict.fromkeys(itemsNames,np.empty((0,0)))

    with open(calibrationCSVPath, "r") as csvfile:
        rowCounter = 0
        rowReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rowReader:
            if(row[0]=="---"):
                rowCounter = 0
            else:
                currentRow = CSVRowToArray(row)
                if(np.shape(Items[itemsNames[rowCounter]]) == (0,0)):
                    Items[itemsNames[rowCounter]] = np.array([currentRow])
                else:
                    Items[itemsNames[rowCounter]] = np.append(Items[itemsNames[rowCounter]],np.array([currentRow]),axis=0)
                rowCounter+=1

    return(Items)
