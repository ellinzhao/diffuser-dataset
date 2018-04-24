
import time
import img

from matplotlib import pyplot as plt

print("making image...")
img_test = img.Img3D(100, 1, 4)

print("making cross section...")
# TODO: add titles to subplots
img_test.x_section()

plt.imshow(img_test.img_filtered[img_test.centers[0][0], :, :], cmap='gray', interpolation='nearest')
plt.savefig("sample_point.png")
plt.close()

start = time.time()
print("generating videos")
#img_test.generate_video(0, False, "z")
#img_test.generate_video(1, False, "y")
#img_test.generate_video(2, False, "x")

#img_test.generate_video(3, True, "z")
#img_test.generate_video(4, True, "y")
#img_test.generate_video(5, True, "x")

end = time.time()
print("time elapsed: %f min" % ((end - start)/60.0))
