
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


class Image:

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

    def gen_2d_points(self):
        arr = np.ones((self.n, self.n))
        centers = []
        num_points = self.k
        while num_points > 0:
            a = np.random.randint(0, self.n)
            b = np.random.randint(0, self.n)
            centers += [(a, b)]
            r1 = np.random.randint(self.r // 2 + 1, self.r)
            # (a, b) is center of point, r1 is radius for point.

            y, x = np.ogrid[-a:self.n - a, -b:self.n - b]
            mask = x * x + y * y <= r1 * r1
            arr[mask] = 255
            num_points -= 1
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
