import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
import gen
import os, subprocess, glob
import matplotlib.cm as cm



class Img2D:
    def __init__(self, n, r, k):
        self.n = n
        self.r = r
        self.k = k
        self.arr, self.centers = self.gen_2d_points(n, r, k)

    def gen_2d_points(self, n, r, k):
        """
        n = dim of the image.
        r = max radius of point.
        k = num of random points.
        all params >= 1.
        """
        arr = np.ones((n, n))
        centers = []
        while k > 0:
            a = np.random.randint(0, n)
            b = np.random.randint(0, n)
            centers += [(a, b)]
            r1 = np.random.randint(r // 2 + 1, r)
            # (a, b) is center of point, r1 is radius for point.

            y, x = np.ogrid[-a:n - a, -b:n - b]
            mask = x * x + y * y <= r1 * r1
            arr[mask] = 255
            k -= 1
        return arr, centers

    def get_img(self):
        # not filtered array
        return self.arr

    def show_img(self):
        img_filtered = ndimage.fourier.fourier_gaussian(self.arr, sigma=2)
        plt.imshow(img_filtered, cmap='gray', interpolation='nearest')

    def get_centers(self):
        return self.centers

    def plt_x_section(self):
        img_filtered = ndimage.filters.gaussian_filter(self.arr, sigma=2, order=0)
        x, y = self.centers[0]
        start = -1
        end = -1
        prev = -1
        prev = 100
        for i in range(0, 500):
            curr = img_filtered[x][i]
            if start == -1 and prev == 1.0 and curr != 1.0:
                start = i
            elif start != -1 and end == -1 and prev != 1.0 and curr == 1.0:
                end = i
                break
            prev = curr
        plt.plot(img_filtered[x][start:end])
        plt.show()


def convolve(n):
    # first build the smoothing kernel

    sigma = 1.7     # width of kernel
    print("making axes...")
    x = np.arange(-n//2,n//2,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-n//2,n//2,1)
    z = np.arange(-n//2,n//2,1)

    print("making meshgrid...")
    xx, yy, zz = np.meshgrid(x, y, z)

    print("making kernel...")
    kernel = np.exp(-(xx**2 + yy**2 +zz**2)/(2*sigma**2))

    # apply to sample data
    #data = np.zeros((n, n))
    #data[n//4, n//4] = 1.

    data = gen.Img3D(n, 2, 1)

    print("convolving...")
    filtered = ndimage.filters.convolve(data.img_unfiltered, kernel, mode="constant")

    return filtered

def low_pass_filter(img):
    """
    Returns a new image that is low pass filtered.
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum, cmap = 'gray')
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2
    w = 8
    fshift[0:crow-w] = 0
    fshift[crow+w:] = 0
    fshift[:, 0:ccol-w] = 0
    fshift[:, ccol+w:] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back
    # plt.imshow(img_back, cmap='gray', interpolation='nearest')

def generate_video(obj, vid=0, plane="xy"):
    # clearing old files
    folder = 'output'
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if file[:5] != "video" and os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    """
    # generating new pngs, mp4
    img = obj
    for i in range(len(img)):
        print(i)
        if plane == "xy":
            plt.imshow(img[i], cmap=cm.Greys_r)
        elif plane == "xz":
            plt.imshow(img[:,i,:], cmap=cm.Greys_r)
        elif plane == "yz":
            plt.imshow(img[:,:,i], cmap=cm.Greys_r)
        plt.savefig('output/file%02d.png' % i)
    os.chdir('output')
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video%d.mp4' % vid
    ])

    # i think this is supposed to remove files but it does not.
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    os.chdir('..')


img_test = convolve(50)

print("generating videos...")
generate_video(img_test, 0, "xy")
generate_video(img_test, 1, "xz")
generate_video(img_test, 2, "yz")


# gradient not so great but that's bc of unfiltered image
# maybe will be better with more pixels
# if not what to do :o
# difference bw convolving with gaussian and the gaussian filter?
