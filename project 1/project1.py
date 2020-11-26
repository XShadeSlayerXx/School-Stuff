import numpy as np
import sklearn as skl
from PIL import Image
import matplotlib.pyplot as plt
import os
from LinearRegression import Regression as rg

# references used: https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
# https://medium.com/machine-learning-researcher/auto-encoder-d942a29c9807

# TODO:
#  file names
#  decode from raw to 3d array (x, y, color)
#  ...
#  read from files into the arrays
#  preprocess the data (including normalize between 0 and 1 or -.5 and .5)
#  matplotlib.pyplot.imshow(np.clip()) to display images
#  .
#  split with skl.train_test_split(data, size, random_state)
#  .
#  .
#  Ideas:
#       arbitrary images, error determined by
#         differences in normalized color of pixels.
#  .
#  parameters:
#       image_shape: size of the image
#       encoded_size: size of the encoded data.
#  .
#  functions:
#       flatten, dense, reshape, input, layers, sequential, model
#       .
#       flatten: reduce np array from 3d to 1d
#       reshape: opposite of flatten, 1d --> 3d
#       dense: final layer of the network, output size is 'encoded_size'
#  .
#  output of encoder goes to the decoder during training, and to a file during testing.
#  .




# todo:
#  order of operations:
#   downscale:
#       get feature map(s) + filter(s) / convolute
#       get pool map
#       flatten
#   upscale:
#       unflatten
#       deconvolute
#       unpool using deconvoute map + pool map
#   error check:
#       flatten original + output
#       compare the two
#   .
#   ML to make the filter maps?
#

def make1D(img : np.array):
    return img.reshape(-1)

def make2D(img : np.array):
    temp = int(img.shape[0]**(1/2))
    return img.reshape(temp,temp)

def make3D(img : np.array, shape):
    return img.reshape(shape)

# takes in a 3d array with z-axis size of 3, returns 3 2d arrays
def splitArray(imgArray : np.array):
    return [x for x in map(np.squeeze, np.split(imgArray, 3, axis = 2))]

def spliceArrays(arrays):
    return np.stack(arrays, axis = 2)

def arbitraryMax(arr):
    if len(arr.shape) > 1:
        return max([arbitraryMax(x) for x in arr[:]])
    return max(arr)

def arbitraryMin(arr):
    if len(arr.shape) > 1:
        return min([arbitraryMin(x) for x in arr[:]])
    return min(arr)

def normBackend(arr, factor):
    if len(arr.shape) > 1:
        return [normBackend(x, factor) for x in arr[:]]
    return arr/factor if factor != 0 else arr

def normalize(arr):
    offset = arbitraryMin(arr)
    if offset < 0:
        arr = arr - offset
    factor = arbitraryMax(arr)
    return normBackend(arr, factor)


class GoodImg:
    def __init__(self, baseImgPath : str = None, baseImgArr : np.array = None):
        self.imgArray = self.setImg(baseImgArr = baseImgArr, baseImgPath = baseImgPath)
        self.shape = self.imgArray.shape
        self.ident = None

    def setImg(self, baseImgPath : str = None, baseImgArr : np.array = None):
        if baseImgPath is not None:
            img = Image.open(baseImgPath)
            return np.array(img)
        else:
            return baseImgArr

    def getImg(self):
        return self.imgArray.reshape(self.shape)

    def __sub__(self, other):
        return self.imgArray - other.imgArray

def getDataFromFolder(folder : str):
    return [GoodImg(folder+'\\'+f) for f in os.listdir(folder) if os.path.isfile(folder+'\\'+f)]

def getIdent(img : GoodImg):
    pass

def matrixAND(m1 : np.array, m2 : np.array):
    if m1.shape != m2.shape:
        raise AssertionError(f'Matrices need to be equivalent shapes: {m1.shape} and {m2.shape}')
    return sum([1 for x in range(m1.shape[0]) for y in range(m1.shape[1]) if m1[x,y] == m2[x,y]])

def getConvoluted(arr : np.array, size : int = -1, filterArr : np.array = None):
    if filterArr is None:
        filterArr = np.random.randint(0,2,(size, size))
    else:
        size = filterArr.shape[0]
    if size%2==0:
        raise AssertionError(f'Convolution size needs to be odd, not {size}')
    sideLength = size//2
    convolArray = np.zeros((arr.shape[0]-sideLength, arr.shape[1]-sideLength))
    for x in range(convolArray.shape[0]-sideLength):
        for y in range(convolArray.shape[1]-sideLength):
            convolArray[x,y] = matrixAND(filterArr, arr[x:x+size,y:y+size])
    return convolArray, filterArr

def applyWeights(arrayReceive : np.array, arrayGive : np.array, offset : tuple):
    for x in range(arrayGive.shape[0]):
        for y in range(arrayGive.shape[1]):
            arrayReceive[x + offset[0], y + offset[1]] += arrayGive[x,y]
    return arrayReceive

# 'convolution transpose'
# TODO: make size a tuple? x,y?
def getDeconvoluted(featureMap : np.array, filterArr : np.array):
    size = filterArr.shape[0]
    sideWidth = size//2
    upsampled = np.zeros((featureMap.shape[0]+sideWidth, featureMap.shape[1]+sideWidth))
    for x in range(featureMap.shape[0]-sideWidth):
        for y in range(featureMap.shape[1]-sideWidth):
            upsampled = applyWeights(upsampled, filterArr, (x,y))
    #TODO: do i need to subtract some value from every index? e.g. the average or 1?
    return upsampled

def maxOf(arr : np.array):
    r = arr.shape[0]
    return max([arr[x,y] for x in range(r) for y in range(r)])

def averageOf(arr : np.array):
    r = arr.shape[0]
    l = len(arr)
    return sum([arr[x,y]/l for x in range(r) for y in range(r)])

#takes in array subsection, returns the subsection reduced to just the max (and max reduced to 1)
def maxLoc(arr : np.array):
    #avg = averageOf(arr)
    max = maxOf(arr) - 1
    return arr - max if max != -1 else arr

def padded(arr : np.array, shape : np.shape):
    temp = np.zeros(shape)
    temp[:arr.shape[0],:arr.shape[1]] = arr
    return temp

#after deconvolution, using the pool map
def unPool(featureMap : np.array, poolMap : np.array):
    size = (featureMap.shape[0]//poolMap.shape[0], featureMap.shape[1]//poolMap.shape[1])
    finalMap = np.zeros(featureMap.shape)
    for x in range(poolMap.shape[0]):
        for y in range(poolMap.shape[1]):
            otx = x * size[0]
            oty = y * size[1]
            tempx = otx + size[0]
            tempy = oty + size[1]
            paddedFeatureMap = padded(featureMap[otx:otx+size[0], oty:oty+size[1]], (size[0],size[1]))
            finalMap[otx:tempx, oty:tempy] += poolMap[x,y] * maxLoc(paddedFeatureMap)
    return finalMap

#TODO: may need to pad the right/lower edges with zeros if shape % size != 0
def getMaxPool(img : np.array, size : int):
    maxPool = np.zeros([x//size for x in img.shape])
    for x in range(0,img.shape[0],size):
        for y in range(0,img.shape[1],size):
            maxPool[x//size, y//size] = maxOf(img[x:x+size,y:y+size])
    return maxPool

def getMSE(img1 : GoodImg, img2 : GoodImg):
    if img1.shape != img2.shape:
        raise AssertionError('Images need to be equivalent shapes')
    errorAbsArray = img1 - img2
    squaredErrorArray = errorAbsArray**2
    MSE = sum(squaredErrorArray)/squaredErrorArray.size
    return MSE

class ImageProcessing:
    def __init__(self, img : GoodImg):
        self.img = img.imgArray
        self.newImg = None
        self.pool = None
        self.convSize = None
        self.filter = None
        self.channels = splitArray(self.img)
        self.channelShape = self.channels[0].shape

#TODO for image in channelShape
    def downsample(self, convSize : int = 3, poolSize : int = 2):
        self.newImg = []
        self.filter = []
        self.convSize = convSize
        for img in self.channels:
            tempImg, tempFilter = getConvoluted(img, convSize)#self.img, convSize)
            self.filter.append(tempFilter)
            self.pool = getMaxPool(img, poolSize)

            self.newImg.append(make1D(tempImg))

    def upsample(self):
        for index in range(len(self.newImg)):
            tmpShape = (self.channelShape[0]-self.convSize//2,self.channelShape[1]-self.convSize//2)
            tempImg = self.newImg[index].reshape(tmpShape)
            tempImg = getDeconvoluted(tempImg, self.filter[index])
            self.newImg[index] = unPool(tempImg, self.pool)
        self.newImg = spliceArrays(self.newImg)

class Network:
    def __init__(self, data, internal_size = None, save_to_file = True):
        inputRegressor = rg('stuff.txt') # output_size = internal_size
        outputRegressor = rg('temp.txt') # output_size = img_dims
        # TODO: calculate error as MSE of output image - input image

def visualizeAdd(what, wherex, wherey, img, title = '', shape = None):
    if len(img.shape) == 1 and shape is None:
        temp = int(img.shape[0]**(1/2))
        img = img.reshape(temp, temp)
    elif shape is not None:
        img = img.reshape(shape)
    img = normalize(img)
    what[wherex][wherey].set_title(title)
    what[wherex][wherey].imshow(img)

if __name__ == '__main__':
    data = getDataFromFolder('data')
    images = []
    for image in data:
        images.append(ImageProcessing(image))
    print(len(images),'images in the set')
    for image in images:
        origShape = image.channelShape
        poolSize = 2
        convSize = 3
        duringShape = (origShape[0]-convSize//2, origShape[1]-convSize//2)
        figure, ax = plt.subplots(nrows = 2, ncols = 3)
        visualizeAdd(ax, 1, 0, image.img, 'pre')
        image.downsample(convSize = convSize, poolSize = poolSize)
        temp = []
        for i, x in enumerate(image.newImg):
            visualizeAdd(ax, 0, i, x, 'RGB during', shape = duringShape)
            temp.append(x.reshape(duringShape))
        visualizeAdd(ax, 1, 1, spliceArrays(temp), 'during combined', shape = (duringShape[0], duringShape[1], 3))
        image.upsample()
        visualizeAdd(ax, 1, 2, image.newImg, 'after', shape = (origShape[0], origShape[1], 3))
        plt.show()
    # autoencoder = Network(data, internal_size = 16, save_to_file = True)
    # autoencoder.train(train_split = .8, learning_rate = .1 )
    # output, error, filepath = autoencoder.predict(GoodImg())
    # filesize = os.path.getsize(filepath)
    # plot old image vs new image + MSE, filesize