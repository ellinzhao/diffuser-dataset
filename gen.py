
import numpy as np
import os, subprocess, glob
from matplotlib import pyplot as plt
from scipy import ndimage, fftpack
import time

import matplotlib.cm as cm


class Img3D:
    def __init__(self, n, r, k):
        self.n = n
        self.r = r
        self.k = k
        self.img_unfiltered, self.centers = self.gen_3d_points()
        # TODO: Might need to change sigma based on n.
        # self.img_filtered = self.img_unfiltered
        self.img_filtered = ndimage.gaussian_filter(self.img_unfiltered, sigma=10)


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
        print("initializing image...")
        arr = np.zeros((self.n, self.n, self.n))
        centers = []
        for _ in range(self.k):
            a = np.random.randint(self.n // 4, 3 * self.n // 4)
            b = np.random.randint(self.n // 4, 3 * self.n // 4)
            c = np.random.randint(self.n//4, 3*self.n//4)
            centers += [(a, b, c)]
            r1 = np.random.randint(self.r // 2, self.r)

            # TODO: fix center + radius selection
            #a, b, c = self.n // 2, self.n // 2, self.n // 2
            #r1 = self.r

            z, y, x = np.ogrid[-c:self.n - c, -b:self.n - b, -a:self.n - a]
            mask = x*x + y*y + z*z <= r1*r1
            arr[mask] = 255
            #arr = ndimage.gaussian_filter(arr, sigma=10)
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


def generate_video_2(img, vid=0, plane="xy"):
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


# Cross section testing
# TODO: make function

#img_test = Img3D(200, 15, 1)
#plt.imshow(img_test.img_filtered[:, 200//2, :], cmap='gray', interpolation='nearest')
#x_section = img_test.img_filtered[:, 200//2, :][:, 200//2]
#plt.plot(range(200), x_section)   #/np.linalg.norm(x_section)
#plt.show()


img_test = Img3D(200, 8, 3)
x, y, z = img_test.centers[0]
#plt.imshow(img_test.img_filtered[:, y, :], cmap='gray', interpolation='nearest')
x_section = img_test.img_filtered[:, y, :][:, x]
plt.plot(range(200), x_section)   #/np.linalg.norm(x_section)
plt.show()

#start = time.time()

#generate_video(img_test, 0, False, "xy")
#generate_video(img_test, 1, False, "xz")
#generate_video(img_test, 2, False, "yz")

print("generating videos")
#generate_video(img_test, 0, True, "xy")
#generate_video(img_test, 1, True, "xz")
#generate_video(img_test, 2, True, "yz")

#generate_video_2(low_pass_filter(img_test.img_unfiltered), 0, "xy")

#end = time.time()
#print("time elapsed: %d" % (end - start))

# TODO: do I want to do this in every direction?

"""
for each center, create subplot:

"""
