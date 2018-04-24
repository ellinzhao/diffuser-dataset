
import numpy as np, matplotlib.cm as cm
import os, subprocess, glob
from matplotlib import pyplot as plt
from scipy import ndimage


class Img3D:

    def __init__(self, n, r, k):
        """
        n = dim of the image.
        r = max radius of point.
        k = num of random points.
        all params >= 1.
        """
        self.n = n
        self.r = r
        self.k = k
        self.img_unfiltered, self.centers = self.gen_3d_points()
        # TODO: Change sigma based on n
        self.img_filtered = ndimage.gaussian_filter(self.img_unfiltered, sigma=6)

    def gen_3d_points(self):
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

            r1 = np.random.randint(self.r // 2, self.r)

            z, y, x = np.ogrid[-c:self.n - c, -b:self.n - b, -a:self.n - a]
            mask = x*x + y*y + z*z <= r1*r1
            arr[mask] = 255
        return arr, centers

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

    def projection(self):
        arr = np.zeros((self.n, self.n))
        for z in range(self.n):
            arr += self.img_unfiltered[z, :, :]
        return arr

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

        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        os.chdir('..')


class Img2D:

    def __init__(self, n, r, k):
        """
        n = dim of the image.
        r = max radius of point.
        k = num of random points.
        all params >= 1.
        """
        self.n = n
        self.r = r
        self.k = k
        self.arr, self.centers = self.gen_2d_points(n, r, k)

    def gen_2d_points(self, n, r, k):
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
