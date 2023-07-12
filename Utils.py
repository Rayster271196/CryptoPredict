import os
import shutil
import numpy as np

# If files exist in pathToCheck then move it to destinationPath
def IfExistsMoveTo(pathToCheck, destinationPath):
    for filename in os.listdir(pathToCheck):
        file_path = os.path.join(pathToCheck, filename)
        destination_path = os.path.join(destinationPath)
        shutil.move(file_path, destination_path)


# Clear files in pathToFolder directory
def clearFilesInDirectory(pathToFolder):
    for filename in os.listdir(pathToFolder):
        file_path = os.path.join(pathToFolder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Returns mean absolute error
def calculate_mae(predicted, actual):
    absolute_errors = np.abs(predicted - actual)
    mae = np.mean(absolute_errors)
    return mae

# Returns mean absolute percentage error
def calculate_mape(predicted, actual):
    absolute_percentage_errors = np.abs((predicted - actual) / actual) * 100
    mape = np.mean(absolute_percentage_errors)
    return mape