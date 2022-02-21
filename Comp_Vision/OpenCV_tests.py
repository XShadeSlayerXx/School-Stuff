import numpy as np
import cv2
from os import walk

SHOW_IMAGES = False
SAVE_IMAGES = True

#import images
path = './sampleImages_Lena/'
images = next(walk(path), (None, None, []))[2]
images = [path+x for x in images]

boxSize = 7
boxFilter = np.ones((boxSize, boxSize))
boxFilter = boxFilter / boxSize ** 2

# gaussian function adapted from:
#  https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (sparklearner)
def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

gaussianFilter = fspecial_gauss(15, 3)

motionSize = 15
motionFilter = np.array([1/motionSize if x == y else 0 for x in range(motionSize) for y in range(motionSize)])

laplacianFilter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

cannyXFilter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
cannyYFilter = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
def cannyCombine(img1, img2):
    final = np.hypot(img1, img2, dtype = float)
    final = final / final.max() * 255
    final = np.where(final > 30, final, 0)
    return final

filters = [boxFilter, gaussianFilter, motionFilter, laplacianFilter, False]
names = ['box', 'gaussian', 'motion', 'laplacian', 'canny']

for count in range(len(images)):
    currentImg = cv2.imread(images[count])
    if SHOW_IMAGES: cv2.imshow(images[count], currentImg)

    for f_count in range(len(filters)):
        filter = filters[f_count]
        if filter is not False:
            resultImg = cv2.filter2D(currentImg, -1, filter)
        else:
            #cannyFilter

            grayImg = currentImg
            #grayImg = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)
            grayImg = cv2.filter2D(grayImg, -1, gaussianFilter)
            resultImg = cv2.filter2D(grayImg, -1, cannyXFilter)
            resultImg2 = cv2.filter2D(grayImg, -1, cannyYFilter)
            resultImg = cannyCombine(resultImg, resultImg2)
        if SHOW_IMAGES: cv2.imshow(names[f_count], resultImg)
        if SAVE_IMAGES:
            slash = images[count].rfind('/')
            dot = images[count].rfind('.')
            name = images[count][slash+1:dot]+ f'-{names[f_count]}.png'
            cv2.imwrite(name, resultImg)


    if SHOW_IMAGES:
        cv2.waitKey()
        cv2.destroyAllWindows()
