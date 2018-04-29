
import numpy as np
from volume import *
import random



# TODO: fix import


def save_outputs(imgs, n):
    """
    :param imgs: list of Img3D objects
    Vectorizes and saves images in .npy files
    """
    data_2d = np.empty((len(imgs), n*n))
    data_3d = np.empty((len(imgs), n*n*n))
    i = 0

    for img in imgs:
        proj = img.projection().flatten()
        np.vstack((data_2d, proj))

        img_arr = img.img_filtered.flatten()
        np.vstack((data_3d, img_arr))

        np.save("labels%i.npy" % i, np.asarray(img.centers))
        i += 1

    # TODO: will it overwrite the old files?
    np.save("data_2D.npy", data_2d)
    np.save("data_3D.npy", data_3d)



def gen_outputs(num_imgs, n, r, percent_range, sigma):

    # making and saving the outputs
    # TODO: throw exceptions if people do not pass in the right parameters
    img_lst = []
    for _ in range(num_imgs):
        img_lst.append(Volume(n, r, percent=random.uniform(percent_range[0], percent_range[1]), sigma_val=sigma))
    save_outputs(img_lst, n)




"""

Interface:

Display outputs to help tune parameters:
- projection
- single 2D centered point
- video of slices

Specify:
- number of images
- density of points
- blur of points
- sigma, can change sigma

"""
