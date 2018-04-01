
import numpy as np
import os, subprocess, glob
from matplotlib import pyplot as plt
from scipy import ndimage
import myfunctions as funcs
import matplotlib.cm as cm


def gen_3d_points(n, r, k):
    """
    n = dim of the image.
    r = max radius of point.
    k = num of random points.
    all params >= 1.
    """
    arr = np.zeros((n, n, n))
    print("done initializing")
    for _ in range(k):
        a = np.random.randint(0, n)
        b = np.random.randint(0, n)
        c = np.random.randint(0, n)
        r1 = np.random.randint(r//2, r)
        print(a, b, c, r1)
        x, y, z = np.ogrid[-a:n-a, -b:n-b, -c:n-c]
        # creating mask with radius r1-i
        mask = x*x + y*y + z*z <= r1*r1
        arr[mask] = 255
        print("done making 1 point")
    return arr


def generate_video(img):
    for i in range(len(img)):
        plt.imshow(img[i], cmap=cm.Greys_r)
        plt.savefig("your_folder/file%02d.png" % i)

    os.chdir("your_folder")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)


vol = gen_3d_points(300, 8, 10)
vol_filtered = ndimage.gaussian_filter(vol, sigma=5, order=0)
generate_video(vol_filtered)
