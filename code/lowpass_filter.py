

import numpy as np
from scipy import fftpack


def low_pass_filter(img):
    """
    Returns a new image that is low pass filtered.
    """
    print("filtering image...")
    f = fftpack.fftn(img)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum, cmap = 'gray')
    x, y, z  = img.shape
    c_x, c_y, c_z = x//2 , y//2, z//2
    w = 10
    # by my logic: z, y, x
    """
    fshift[0:c_z-w, :, :] = 0
    fshift[c_z+w:, :, :] = 0

    fshift[:, 0:c_y-w, :] = 0
    fshift[:, c_y+w:, :] = 0

    fshift[:, :, 0:c_x - w] = 0
    fshift[:, :, c_x+z:] = 0
    """
    z, y, x = np.ogrid[-c_z:z-c_z, -c_y:x-c_y, -c_x:x-c_x]
    mask = x*x + y*y + z*z >= w**2
    fshift[mask] = 0


    f_ishift = fftpack.ifftshift(fshift)
    img_back = fftpack.ifftn(f_ishift)
    img_back = np.abs(img_back)
    print("done filtering")
    return img_back
    # plt.imshow(img_back, cmap='gray', interpolation='nearest')


