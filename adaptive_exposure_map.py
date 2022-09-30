import numpy as np
import cv2
from guidedFilter import GuidedFilter


def adaptiveExposureMap(img, restored):
    # Constants
    Lambda = 0.3

    img = np.uint8(img)
    restored = np.uint8(restored)
    
    # Changing color space of images from BGR to luma-chroma
    Yi_rb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Yj_rb = cv2.cvtColor(restored, cv2.COLOR_BGR2YCrCb)

    # Normalizing images
    norm_Yi_rb = (Yi_rb-Yi_rb.min()) / (Yi_rb.max()-Yi_rb.min())
    norm_Yj_rb = (Yj_rb-Yj_rb.min()) / (Yj_rb.max()-Yj_rb.min())

    # Yi = illumination intensity of input image
    # Yj = illumination intensity of restored image
    Yi = norm_Yi_rb[:,:,0]
    Yj = norm_Yj_rb[:,:,0]

    # Clipping the array values to [0.01, 1] 
    Min = 10**-2
    Max = 1
    Yi = np.clip(Yi,Min,Max)
    Yj = np.clip(Yj,Min,Max)

    # S = (Yj*Yi + lambda*Yi^2)/(Yj^2 + lambda*Yi^2)
    S = (Yj*Yi + Lambda*(Yi**2)) / ((Yj**2) + Lambda*(Yi**2))

    # Parameters for guided filter
    radius = 50
    eps = pow(10,-3)
    myfilter = GuidedFilter(Yi_rb, radius, eps)

    # Filtering S using Yi_rb as guidance image
    filtered_S = myfilter.filter(S)

    # adaptive exposure map S(x) = GuidedFilter[S]
    S_x = np.zeros(img.shape)
    S_x[:,:,0] = filtered_S[0]
    S_x[:,:,1] = filtered_S[1]
    S_x[:,:,2] = filtered_S[2]

    return S_x


def applyAdaptiveMap(restored, S_x):

    # Constants
    Min = 0
    Max = 255

    # Changing datatype of restored image to float
    if restored.dtype != np.float64:
        restored = np.float64(restored)

    # Applying Adaptive map filter
    restored = restored * S_x

    # Clipping restored image values to [0,255]
    restored = np.clip(restored, Min, Max)

    # Changing datatype back to uint8
    restored = np.uint8(restored)

    return restored