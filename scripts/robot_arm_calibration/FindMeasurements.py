import numpy as np
import yaml
import sys
import csv
import os

from robot_arm_calibration.AnalyticGeometricModel import JOINTS_NUMBER, EEItemsNames
from robot_arm_calibration.CalibrationCSVParsing import calibrationCSVParsing

if __name__ == '__main__':

    ### Load configurations
    configurations = np.empty((0,0))
    print("Loading target configurations...")
    try:
        with open("TargetConfigurations.yaml","r") as file:
                configurations = np.array(yaml.load(file, Loader=yaml.FullLoader)["targetConfigurations"]).reshape((-1,JOINTS_NUMBER))
    except:
        print("No valid configurations file ! Are you in a proper calibrated model folder ?")
        sys.exit(-1)
    print("Done !")

    ### Load measurements
    measurementsPath = sys.argv[1]

    print("Retrieving calibration measurement data...")
    itemsNames = EEItemsNames + ["jointsPositions"]
    try:
        Items = calibrationCSVParsing(itemsNames, measurementsPath)
    except:
        print(
            "Invalid calibration measurements CSV !"
        )
        sys.exit(-1)
    print("Done !")

    ### Find measurements
    MeasurementsIndex = [np.argmin(np.linalg.norm(Items["jointsPositions"] - configuration, axis=1)) for configuration in configurations]

    ### Build new measurements
    os.mkdir("measurements")
    with open("measurements/Calibration.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for index in MeasurementsIndex:
            for item in Items.items():
                writer.writerow(item[1][index])
            writer.writerow(["---"])
