
from volume import *
import random


def save_outputs(imgs, n):
    """
    :param imgs: list of Volume objects
    :param n: size of object (n x n x n)
    Vectorizes and saves images in .npy files
    """
    data_2d = np.empty((len(imgs), n*n))
    data_3d = np.empty((len(imgs), n*n*n))
    i = 0
    for img in imgs:
        proj = img.projection().flatten()
        data_2d[i, :] = proj

        img_arr = img.vol_filtered.flatten()
        data_3d[i, :] = img_arr

        np.save("out_labels/labels%i.npy" % i, np.asarray(img.centers))
        i += 1
    np.save("out/data_2D.npy", data_2d)
    np.save("out/data_3D.npy", data_3d)


def gen_outputs(num_imgs=10, n=100, r=1, percent_range=(0.1, 0.2), sigma=5):
    """
    :param num_imgs: number of images with the following params...
    :param n: size of volume (n x n x n)
    :param r: radii of points will be in [r//2, r]
    :param percent_range: tuple representing range of densities
    :param sigma: sigma of Gaussian
    """
    img_lst = []
    for _ in range(num_imgs):
        img_lst.append(Volume(n, r, percent=random.uniform(percent_range[0], percent_range[1]), sigma=sigma))
    save_outputs(img_lst, n)


# Testing volumes
vol = Volume(100, 1, 0.15, 5)
vol.show_projection()
vol.x_section()
vol.test_sigma_img(8)
# vol.test_sigma_video(8)


# Saving volumes to file
gen_outputs(10, 100, 1, (0.1, 0.15), 5)

