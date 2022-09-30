import numpy as np

#J_GB = recovered blue and green channels
def correctRChannel(img, J_GB):
    img = np.float32(img)
    J_GB = np.float32(J_GB)
    J_GBR = J_GB.copy()
    
    #normalized avg value of recovered green channel
    avgGr = np.mean(J_GB[:, :, 0])/255
    #normalized avg value of recovered blue channel
    avgBr = np.mean(J_GB[:, :, 1])/255

    avgRr = 1.5 - avgGr - avgBr

    og_red = img[:, :, 2]
    #normalized average value of original red channel
    avgR = np.mean(og_red)/255

    #compensation coefficient delta
    delta = avgRr/avgR

    #recovered red channel
    Rrec = og_red * delta
    Rrec = (Rrec - Rrec.min()) / (Rrec.max() - Rrec.min())
    Rrec = Rrec * 255   #scale back to 255
    Rrec = np.clip(Rrec, 0, 255)
    J_GBR[:, :, 2] = Rrec
    J_GBR = np.uint8(J_GBR) #check!

    return J_GBR
