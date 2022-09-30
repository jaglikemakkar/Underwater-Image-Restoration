import numpy as np
import cv2
#from imageio import imread

#find D(x) = largest differnce among three different channels

#img = imread("/home/bhumika/CS517-DIPA/underwater-image-restoration/sample_images/BUL_T1A_0028.jpg")

#data type to compare intensity values and get corresponding argument
class arg_val:
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f


#get maximum channel out of blue and green
def maxChannel(img):
    m, n = (img.shape[0], img.shape[1])
    maxChannelImg = np.zeros((m, n), dtype = np.float32)
    for i in range(m):
        for j in range(n):
            maxChannelImg[i,j] = max(img.item((i, j, 0)), img.item((i, j, 1)))
    return maxChannelImg

def getMaxChannelLocal(img, window):
    # m = height, n = width
    m, n = (img.shape[0], img.shape[1])
    imgChannel = np.zeros((m, n), dtype = np.float16)

    #pad the image to get a modified size to accomodate patches
    padSize = int((window - 1)/2)
    h = m+window-1
    w = n+window-1
    imgChannelTemp = np.zeros((h, w))

    #copy the img as initialization
    imgChannelTemp[:, :] = 0
    imgChannelTemp[padSize:h-padSize, padSize:w-padSize] = img

    #find local max channel
    #localMax = 0
    for i in range(padSize, h-padSize):
        for j in range(padSize, w-padSize):
            localMax = 0
            for x in range(i-padSize, i+padSize+1):
                for y in range(j-padSize, j+padSize+1):
                    localMax = max(localMax, imgChannelTemp.item((x, y)))
            imgChannel[i-padSize, j-padSize] = localMax
    
    return imgChannel

def getLargestDiff(img, window):
    imgN = img/255  #check why normalization here
    greater_of_GB = maxChannel(imgN)
    max_GB = getMaxChannelLocal(greater_of_GB, window)
    redChannel = imgN[:,:,2]
    max_R = getMaxChannelLocal(redChannel, window)
    return max_R - max_GB

# B_c=avg(I_c(arg min_x D(x))),c∈{g, b}
def backgroundLight(img, D):
    img = np.float32(img)
    h = D.shape[0]
    w = D.shape[1]

    #append image coordinates, intensity value
    arg_val_list = []
    for i in range(h):
        for j in range(w):
            temp = arg_val(i, j, D[i, j])
            arg_val_list.append(temp)

    #sort arg_list to get the argument corresponding to min D
    arg_val_list = sorted(arg_val_list, key = lambda arg_val: arg_val.f)
    #avg over blue and green channels
    min_pair = arg_val_list[0]
    B = np.mean([img[min_pair.x, min_pair.y, 0], img[min_pair.x, min_pair.y, 1]])

    #required later
    B_GB = img[min_pair.x, min_pair.y, 0:2]
    B_RGB = img[min_pair.x, min_pair.y, :]

    return B, B_GB, B_RGB

