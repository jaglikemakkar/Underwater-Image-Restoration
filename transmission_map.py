import math
import numpy as np
from guidedFilter import GuidedFilter

#coarse estimation of the transmission map, based on Dark Channel Prior Theory assumptions
def estimate_t(I, B, window):
    temp = I/B
    padSize = math.floor(window/2)
    paddedTemp = np.pad(temp, ((padSize, padSize), (padSize, padSize), (0,0)) , 'constant')

    #initialize t
    t = np.zeros((I.shape[0], I.shape[1], 2))

    #t(x)=1−minc∈{g,b}(minx∈Ω(Ic(x)Bc))
    #todo--- doubt!
    for i, j in np.ndindex(I.shape[0], I.shape[1]):
        t[i, j, 0] = 1 - np.min(paddedTemp[i:i+window, j:j+window, 0])      #t_blue
        t[i, j, 1] = 1 - np.min(paddedTemp[i:i+window, j:j+window, 1])      #t_green
        
        #according to me, as medium transmission maps of b and g channels are assumed to be identical
        #tmp = min(np.min(paddedTemp[i:i+window, j:j+window, 0]), np.min(paddedTemp[i:i+window, j:j+window, 1]))
        #t[i, j, 0] = 1 - tmp
        #t[i, j, 1] = 1 - tmp

    return t
    

# refine the transmission map using guided filter
def refine_t(t, img):

    # parameters for guided filter
    radius = 50
    eps = pow(10,-3)

    # creating object of class GuidedFilter

    myfilter = GuidedFilter(img, radius, eps)
    t0 = t[:,:,0]
    t1 = t[:,:,1]

    # Filtering green, blue channels
    t[:,:,0] = myfilter.filter(t0)
    t[:,:,1] = myfilter.filter(t1)

    # clipping the values to [0.1, 0.9]
    Min = 0.1
    Max = 0.9
    t = np.clip(t, Min, Max)

    return t