import numpy
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import math
import pickle

## Encode array and writes file.
## Header contains coeff_slices and coeff_shapes.
## Then arr encoded
## 2.0 will contain shape in header
def huffmanEncode(arr):
    arrToEncode, coeff_slices, coeff_shapes = pywt.ravel_coeffs(arr)
    output = []
    zeroCount = 0
    for num in arrToEncode:
        if round(num) == 0:
            zeroCount += 1
        else:
            if zeroCount == 0:
                output.append(num)
            else:
                output.append((0, zeroCount))
                output.append(num)
                zeroCount = 0
    ## Add final 0s
    if zeroCount != 0:
        output.append((0, zeroCount))
    outArr = numpy.array(output)
    writeEncoded(coeff_slices, coeff_shapes, outArr, "Compressed.p")

def huffmanDecode():
    slice, shape, compressedArr = loadEncoded("Compressed.p")
    decompressedArr = []
    for num in compressedArr:
        if type(num) == tuple:
            for i in range(num[1]):
                decompressedArr.append(0)
        else:
            decompressedArr.append(num)
    outArr = numpy.array(decompressedArr)
    output = pywt.unravel_coeffs(outArr, slice, shape, output_format='wavedec2')
    return output


def writeEncoded(slice, shape, arr, filename):
    f = open(filename, 'wb')
    pickle.dump(slice, f)
    pickle.dump(shape, f)
    pickle.dump(arr, f)
    f.close()

def loadEncoded(filename):
    f = open(filename, 'rb')
    objs = []
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    return objs[0], objs[1], objs[2]

print("Packages Imported")
plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join('dog.jpg'))
B = np.mean(A, -1)  # Convert RGB to grayscale

## Find best wavelet
bestWave = ''
bestWaveValue = 999999
wavesToCompare = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
keep = .1
for w in wavesToCompare:
    coeffs = pywt.wavedec2(B, wavelet=w)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

    thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind
    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
    RMSE = np.square(np.subtract(Arecon[:len(B), :len(B[1])], B)).mean()
    if abs(RMSE) < bestWaveValue:
        bestWave = w
        bestWaveValue = abs(RMSE)

print("--")
print(bestWave)
print(bestWaveValue)

## Wavelet Compression
w = bestWave
coeffs = pywt.wavedec2(B, wavelet=w)
coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
ind = np.abs(coeff_arr) > thresh
Cfilt = coeff_arr * ind  # Threshold small indices
coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

#Compress/Decompress
huffmanEncode(coeffs_filt)
decompressed = huffmanDecode()

# Plot reconstruction
Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
plt.figure()
plt.imshow(Arecon.astype('uint8'), cmap='gray')
plt.axis('off')
plt.title('keep = ' + str(keep))
plt.show()
