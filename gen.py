
import numpy as np
import os, subprocess, glob
from matplotlib import pyplot as plt
from scipy import ndimage
import time

import matplotlib.cm as cm


class Img3D:
    def __init__(self, n, r, k):
        self.n = n
        self.r = r
        self.k = k
        self.img_unfiltered, self.centers = self.gen_3d_points()
        # Might need to change sigma based on n.
        self.img_filtered = self.filter()


    def filter(self):
        filtered = np.empty((self.n, self.n, self.n))
        for i in range(self.n):
            filtered[i, :, :] = ndimage.gaussian_filter(self.img_unfiltered[i, :, :], sigma=4)
        return filtered


    def gen_3d_points(self):
        """
        n = dim of the image.
        r = max radius of point.
        k = num of random points.
        all params >= 1.
        """
        arr = np.zeros((self.n, self.n, self.n))
        centers = []
        for _ in range(self.k):
            a = np.random.randint(self.n // 4, 3 * self.n // 4)
            b = np.random.randint(self.n // 4, 3 * self.n // 4)
            c = np.random.randint(self.n//4, 3*self.n//4)
            centers += [(a, b, c)]
            r1 = np.random.randint(self.r // 2, self.r)

            a, b, c = self.n // 2, self.n // 2, self.n // 2
            r1 = self.r

            z, y, x = np.ogrid[-a:self.n - a, -b:self.n - b, -c:self.n - c]
            mask = x*x + y*y + z*z <= r1*r1
            arr[mask] = 255
        return arr, centers

    def show_img(self, filter_flag):
        if filter_flag == 0:
            plt.imshow(self.img_unfiltered[self.centers[0][0]], cmap='gray', interpolation='nearest')
        elif filter_flag == 1:
            plt.imshow(self.img_filtered[:, self.centers[0][0], :, :], cmap='gray', interpolation='nearest')


def generate_video(obj, vid=0, filtered=True, plane="xy"):
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
    if filtered:
        img = obj.img_filtered
    else:
        img = obj.img_unfiltered

    # generating new pngs, mp4
    for i in range(len(img)):
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


#img_test = Img3D(30, 5, 1)


#start = time.time()

#generate_video(img_test, 0, False, "xy")
#generate_video(img_test, 1, False, "xz")
#generate_video(img_test, 2, False, "yz")

#generate_video(img_test, 3, True, "xy")
#generate_video(img_test, 4, True, "xz")
#generate_video(img_test, 5, True, "yz")

#end = time.time()
#print(end - start)
