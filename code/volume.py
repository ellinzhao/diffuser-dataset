
import numpy as np, matplotlib.cm as cm
import os, subprocess, glob
from matplotlib import pyplot as plt
from scipy import ndimage


class Volume:

    def __init__(self, n, r, percent=0.30, sigma=6):
        """
        :param n: dim of image (n x n x n)
        :param r: radius of points
        :param percent: density of points
        :param sigma: sigma of Gaussian filter
        """
        self.n = n
        self.r = r
        self.percent = percent
        self.sigma = sigma
        self.point_unfiltered = self.gen_single_point()
        self.point_filtered = ndimage.gaussian_filter(self.point_unfiltered, sigma=self.sigma)
        self.set_k()
        self.vol_unfiltered, self.centers = self.gen_points()
        self.vol_filtered = ndimage.gaussian_filter(self.vol_unfiltered, sigma=self.sigma)

    def gen_single_point(self):
        arr = np.zeros((self.n, self.n, self.n))
        a = self.n//2
        z, y, x = np.ogrid[-a:self.n - a, -a:self.n - a, -a:self.n - a]
        mask = x*x + y*y + z*z <= self.r**2
        arr[mask] = 255
        return arr

    def gen_points(self):
        # TODO: add timeout error if radius and density params are infeasible
        print("Initializing volume...")
        arr = np.zeros((self.n, self.n, self.n))
        centers = []
        for i in range(self.k):
            in_radius = [True]
            while any(in_radius):
                a = np.random.randint(4*self.r, self.n - 4*self.r)
                b = np.random.randint(4*self.r, self.n - 4*self.r)
                c = np.random.randint(4*self.r, self.n - 4*self.r)
                in_radius = [(a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2 <= 4*self.r ** 2 for z, y, x in centers]
            centers.append((c, b, a))
            r1 = np.random.randint(self.r//2, self.r)
            z, y, x = np.ogrid[-c:self.n - c, -b:self.n - b, -a:self.n - a]
            mask = x*x + y*y + z*z <= r1*r1
            arr[mask] = 255
        return arr, centers

    def test_sigma_img(self, sigma_val):
        new_vol = ndimage.gaussian_filter(self.point_unfiltered, sigma=sigma_val)
        plt.imshow(new_vol[self.n//2, :, :], cmap='gray', interpolation='nearest')
        plt.savefig("test_sigma_%f.png" % sigma_val)
        plt.close()

    def test_sigma_video(self, sigma_val):
        new_vol = ndimage.gaussian_filter(self.vol_unfiltered, sigma=sigma_val)
        self.generate_video(new_vol, vid="test_sigma_%f" % sigma_val)

    def show_projection(self):
        # if you want to change density, prob makes more sense to make a new volume
        plt.imshow(self.projection(), cmap='gray', interpolation='nearest')
        plt.savefig("test_density_%f.png" % self.percent)
        plt.close()

    def set_k(self):
        # TODO: change to 3db point?
        row = self.point_filtered[self.n//2, self.n//2, :]
        last = 0
        start = 0
        end = self.n
        for i in range(self.n):
            if last == 0 and row[i] != 0:
                start = i
            elif last != 0 and row[i] == 0:
                end = i
            last = row[i]
        point_r = (end - start) / 2.0   # radius of blurred point
        self.r_filtered = point_r
        self.k = int(round(self.percent * self.n**3 / (4.0 / 3 * 3.14159 * point_r**3)))

    def x_section(self):
        fig = plt.figure(figsize=(12, 4 * len(self.centers)))
        # TODO: change title
        offset = 5*self.r_filtered//4
        rows = len(self.centers)
        for i in range(rows):
            z, y, x = self.centers[i]

            z_vals = self.vol_filtered[:, y, x]
            z_vals = z_vals[max(0, z - offset): min(self.n, z + offset)]
            fig.add_subplot(rows, 3, 3 * i + 1)
            # ax_z.set_title("")
            plt.plot(np.arange(0, len(z_vals), 1), z_vals)

            y_vals = self.vol_filtered[z, :, x]
            y_vals = y_vals[max(0, y - offset): min(self.n, y + offset)]
            fig.add_subplot(rows, 3, 3 * i + 2)
            plt.plot(np.arange(0, len(y_vals), 1), y_vals)

            x_vals = self.vol_filtered[z, y, :]
            x_vals = x_vals[max(0, x - offset): min(self.n, x + offset)]
            fig.add_subplot(rows, 3, 3 * i + 3)
            plt.plot(np.arange(0, len(x_vals), 1), x_vals)

        plt.savefig("cross_section.png")
        plt.close()

    def projection(self):
        arr = np.zeros((self.n, self.n))
        for z in range(self.n):
            arr += self.vol_filtered[z, :, :]
        return arr

    def generate_video(self, arr, vid="video", plane="z"):
        # clearing old files
        folder = 'out_videos_videos'
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
        for i in range(len(arr)):
            #workdone = i/float(len(arr))
            # TODO: print statement broke
            #print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
            if plane == "z":
                plt.imshow(arr[i], cmap=cm.Greys_r)
            elif plane == "y":
                plt.imshow(arr[:,i,:], cmap=cm.Greys_r)
            elif plane == "x":
                plt.imshow(arr[:,:,i], cmap=cm.Greys_r)
            plt.savefig('out_videos/file%02d.png' % i)
        os.chdir('out_videos')
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            '%s.mp4' % vid
        ])

        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        os.chdir('..')

