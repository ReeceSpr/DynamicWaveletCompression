from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import math

## Python Script used to demonstrate the use of alternative wavelet decomposition in images.
## CODE GIT: https://github.com/dynamicslab/databook_python/blob/master/CH02/CH02_SEC06_4_Wavelet.ipynb
## CODE GIT: https://github.com/dynamicslab/databook_python/blob/master/CH02/CH02_SEC06_5_WaveletCompress.ipynb
## Wave Origin: https://pywavelets.readthedocs.io/en/latest/install.html


print("Packages Imported")
plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join('dog.jpg'))
B = np.mean(A, -1)  # Convert RGB to grayscale

## Wavelet decomposition (2 level)
n = 1
w = 'db1'
coeffs = pywt.wavedec2(B, wavelet=w, level=n)

# normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)
plt.imshow(arr, cmap='gray', vmin=-0.25, vmax=0.75)
plt.show()

### Wavelet Compression
## = 2
## = 'db1'
##oeffs = pywt.wavedec2(B, wavelet=w, level=n)
##
##oeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
##
##sort = np.sort(np.abs(coeff_arr.reshape(-1)))
##
##or keep in (0.1, 0.05, 0.01, 0.005):
##   thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
##   ind = np.abs(coeff_arr) > thresh
##   Cfilt = coeff_arr * ind  # Threshold small indices
##
##   coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')
##
##   # Plot reconstruction
##   Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
##   plt.figure()
##   plt.imshow(Arecon.astype('uint8'), cmap='gray')
##   plt.axis('off')
##   plt.title('keep = ' + str(keep))
##   plt.show()
##
##
##