import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import gen
import os, subprocess, glob
import matplotlib.cm as cm



def convolve(n):
    # first build the smoothing kernel

    sigma = 2     # width of kernel
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

    data = gen.Img3D(200, 15, 1)

    print("convolving...")
    filtered = ndimage.filters.convolve(data.img_unfiltered, kernel, mode="constant")

    return filtered


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


img_test = convolve(200)

print("generating videos...")
generate_video(img_test, 0, "xy")
#generate_video(img_test, 1, "xz")
#generate_video(img_test, 2, "yz")


# gradient not so great but that's bc of unfiltered image
# maybe will be better with more pixels
# if not what to do :o
# difference bw convolving with gaussian and the gaussian filter?
