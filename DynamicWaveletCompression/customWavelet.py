from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import math


# c = math.sqrt(2) / 2
# dec_lo, dec_hi, rec_lo, rec_hi = [c, c], [-c, c], [c, c], [c, -c]
# filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
# myWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)


class HaarFilterBank(object):
    @property
    def filter_bank(self):
        c = math.sqrt(2) / 2
        dec_lo, dec_hi, rec_lo, rec_hi = [c, c], [-c, c], [c, c], [c, -c]
        return [dec_lo, dec_hi, rec_lo, rec_hi]


filter_bank = HaarFilterBank()
myOtherWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)
print(myOtherWavelet)

A = imread(os.path.join('dog.jpg'))
B = np.mean(A, -1)
n = 2
coeffs = pywt.wavedec2(B, wavelet=myOtherWavelet, level=n)

# normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

plt.imshow(arr, cmap='gray', vmin=-0.25, vmax=0.75)
plt.show()

## Wavelet Compression
coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

for keep in (0.1, 0.05, 0.01, 0.005):
    thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind  # Threshold small indices

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

    # Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=myOtherWavelet)
    plt.figure()
    plt.imshow(Arecon.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.title('keep = ' + str(keep))
    plt.show()
print("Script Complete")
