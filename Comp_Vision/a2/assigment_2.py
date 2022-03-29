import cv2
from matplotlib import pyplot as plt
import numpy as np

# k-means clusters
clusters = 5

SHOW_IMAGES = False
SHOW_HISTOGRAMS = False

SAVE_FILES = False

images = [
    r'sample.jpg',
    r'sample2.jpg'
]

class grey:
    def __init__(self, name, image):
        self.name = name
        self.image = image

gray = [
    # grey(x, cv2.imread(x)) for x in images
    grey(x, cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in images
]

if SHOW_IMAGES:
    for img in gray:
        cv2.imshow(img.name, img.image)

#display the histogram + images
if SHOW_HISTOGRAMS:
    for img in gray:
        flat = img.image.reshape((-1, 1))
        plt.figure()
        plt.title(img.name)
        plt.xlabel('Pixel Value')
        plt.ylabel('Amount')
        plt.hist(flat, 256)
        if SAVE_FILES: plt.savefig(f'{img.name[:-4]}-histogram.png')
        plt.show()

if SHOW_IMAGES or SHOW_HISTOGRAMS:
    cv2.waitKey(0)

# k-means clustering first
for index, image in enumerate(gray):
    Z = image.image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back into uint8 and reform image
    center = np.uint8(center)
    res = center[label.flatten()]
    result = res.reshape((image.image.shape))
    if SAVE_FILES: cv2.imwrite(f'{image.name[:-4]}-gray-kmeans-{clusters}.png',result)
    if SHOW_IMAGES: cv2.imshow(image.name, result)
    gray[index] = grey(image.name, result)

# otherwise use edge/contour detection
for image in gray:
    # get the image thresholds
    _, threshold = cv2.threshold(image.image, np.mean(image.image), 255, cv2.THRESH_BINARY_INV)
    # apply the canny filter to get the edges
    edges = cv2.dilate(cv2.Canny(threshold, 0, 255), None)

    if SAVE_FILES: cv2.imwrite(f'{image.name[:-4]}-edge.png', edges)
    if SHOW_IMAGES: cv2.imshow(image.name, edges)

cv2.waitKey(0)
cv2.destroyAllWindows()