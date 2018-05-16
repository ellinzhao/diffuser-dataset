
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


def gen_outputs(num_imgs, n, r, percent_range, sigma):
    img_lst = []
    for _ in range(num_imgs):
        img_lst.append(Volume(n, r, percent=random.uniform(percent_range[0], percent_range[1]), sigma=sigma))
    save_outputs(img_lst, n)


gen_outputs(10, 100, 1, (0.1, 0.15), 5)

