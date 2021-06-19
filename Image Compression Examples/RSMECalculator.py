# Takes an two images and computes the RMSE.
# Prints to console.

import sys
from matplotlib.image import imread
import numpy as np
import os

if len(sys.argv) != 3:
    print("Incorrect arguments given.")
    exit(1)

A = imread(os.path.join(sys.argv[1]))
A = np.mean(A, -1)  # Convert RGB to grayscale

B = imread(os.path.join(sys.argv[2]))
B = np.mean(B, -1)  # Convert RGB to grayscale

# Calculate RMSE
RMSE = np.square(np.subtract(A, B)).mean()

print(RMSE)