import numpy as np
import cv2
from background_light import getLargestDiff, backgroundLight
from transmission_map import estimate_t, refine_t

#Jc(x)=Ic(x)−Bc/tc(x)+Bc,c∈{g, b}
def getRestoredChannel(I, t, B):
    I = np.float32(I)
    J = np.zeros(I.shape, dtype = np.float32)
    for c in range(2):  #for b and g
        J[:, :, c] = (I[:, :, c]-B[c])/t[:, :, c]
        J[:, :, c] = J[:, :, c] + B[c]
    
    #scale J to (0, 255) and astype = uint8(check!)
    J = (J-J.min())/(J.max()-J.min())*255
    J = np.uint8(np.clip(J, 0, 255))
    return J


def GBDehaze(img, window, resultPath, prefix):
    print("Calculating Largest Difference between channels...")
    largestDiff = getLargestDiff(img, window)
    print("Getting Background Light...")
    B, B_GB, B_RGB = backgroundLight(img, largestDiff)
    print("Estimating medium transmission map...")
    t_map = estimate_t(img, B_RGB, window)
    
    cv2.imwrite(resultPath +  '\\' + prefix + '_transmission.jpg', np.uint8(t_map[:, :, 0] * 255))

    print("Refining transmission map... ")
    t_map = refine_t(t_map, img)
    cv2.imwrite(resultPath +  '\\' + prefix + '_refined_transmission.jpg', np.uint8(t_map[:, :, 0] * 255))

    print("Restoring GB channels...")
    recovered_gb = getRestoredChannel(img, t_map, B_RGB)
    return recovered_gb

