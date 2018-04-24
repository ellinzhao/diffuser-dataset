
import numpy as np
import .img


def data_2d(imgs):
    data_2d = np.empty(0)
    data_3d = np.empty(0)
    labels = np.empty(0)

    for img in imgs:
        proj = img.projection()
        proj.flatten()
        np.vstack((data_2d, proj))

        img_arr = img.img_filtered
        img_arr.flatten()
        np.vstack((data_3d, img_arr))

        np.vstack((labels, np.asarray(img.centers)))

    np.save("data_2D.npy", data_2d)
    np.save("data_3D.npy", data_3d)
    np.save("labels.npy", labels)


