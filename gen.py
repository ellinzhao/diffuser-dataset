
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
        self.img_filtered = ndimage.gaussian_filter(self.img_unfiltered, sigma=11)

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
        for i in range(self.k):
            in_radius = [True]
            while any(in_radius):
                a = np.random.randint(4*self.r, self.n - 4*self.r)
                b = np.random.randint(4*self.r, self.n - 4*self.r)
                c = np.random.randint(4*self.r, self.n - 4*self.r)
                in_radius = [(a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2 <= 4*self.r ** 2 for x, y, z in centers]
            centers.append((c, b, a))

            #r1 = np.random.randint(self.r // 2, self.r)
            r1 = self.r

            # TODO: fix center + radius selection
            #a, b, c = self.n // 2, self.n // 2, self.n // 2

            z, y, x = np.ogrid[-c:self.n - c, -b:self.n - b, -a:self.n - a]
            mask = x*x + y*y + z*z <= r1*r1
            arr[mask] = 255
        return arr, centers

    def show_img(self, filter_flag):
        if filter_flag == 0:
            plt.imshow(self.img_unfiltered[self.centers[0][0]], cmap='gray', interpolation='nearest')
        elif filter_flag == 1:
            plt.imshow(self.img_filtered[:, self.centers[0][0], :, :], cmap='gray', interpolation='nearest')

    def x_section(self):
        fig = plt.figure(figsize=(12, 4 * len(self.centers)))
        offset = 40
        rows = len(self.centers)
        for i in range(rows):
            z, y, x = self.centers[i]
            print((z,y,x))

            z_vals = self.img_filtered[:, y, x]
            z_vals = z_vals[max(0, z - offset*self.r): min(self.n, z + offset*self.r)]
            fig.add_subplot(rows, 3, 3 * i + 1)
            plt.plot(np.arange(0, len(z_vals), 1), z_vals, marker='o')

            y_vals = self.img_filtered[z, :, x]
            y_vals = y_vals[max(0, y - offset*self.r): min(self.n, y + offset*self.r)]
            fig.add_subplot(rows, 3, 3 * i + 2)
            plt.plot(np.arange(0, len(y_vals), 1), y_vals, marker='o')

            x_vals = self.img_filtered[z, y, :]
            x_vals = x_vals[max(0, x - offset*self.r): min(self.n, x + offset*self.r)]
            fig.add_subplot(rows, 3, 3 * i + 3)
            plt.plot(np.arange(0, len(x_vals), 1), x_vals, marker='o')

        plt.savefig("cross_section.png")
        plt.close()

    def generate_video(self, vid=0, filtered=True, plane="z"):
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
            img = self.img_filtered
        else:
            img = self.img_unfiltered

        # generating new pngs, mp4
        for i in range(len(img)):
            workdone = i/float(len(img))
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
            if plane == "z":
                plt.imshow(img[i], cmap=cm.Greys_r)
            elif plane == "y":
                plt.imshow(img[:,i,:], cmap=cm.Greys_r)
            elif plane == "x":
                plt.imshow(img[:,:,i], cmap=cm.Greys_r)
            plt.savefig('output/file%02d.png' % i)
        os.chdir('output')
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'video%d.mp4' % vid
        ])

        # I think this is supposed to remove files but it does not.
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



img_test = Img3D(500, 1, 9)

print("making cross section...")
img_test.x_section()


start = time.time()

#img_test.generate_video(0, False, "z")
#img_test.generate_video(1, False, "y")
#img_test.generate_video(2, False, "x")

print("generating videos")
img_test.generate_video(3, True, "z")
img_test.generate_video(4, True, "y")
img_test.generate_video(5, True, "x")

end = time.time()

print("time elapsed: %f min" % ((end - start)/60.0))

