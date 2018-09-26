def anisodiff(im, niter=12, delta=1.0/7.0, kappa=11):
    import numpy as np
    from scipy import misc, ndimage

    # convert input image
    im = im.astype('float64')

    # initial condition
    u = im

    # center pixel distances
    dx = 1
    dy = 1
    dd = np.sqrt(2)

    # 2D finite difference windows
    hN = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64) #N
    hS = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64) #S
    hE = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64) #E
    hW = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64) #W
    hNE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64) #NE
    hSE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64) #SE
    hSW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64) #SW
    hNW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64) #NW

    masks = [hN, hS, hE, hW, hNE, hSE, hSW, hNW]
   

    for r in range(niter):

        # taking dx=dy=1 and dd=sqrt(2)
        PM = sum([np.multiply(np.exp(-np.square(ndimage.filters.convolve(u, m)/kappa)), ndimage.filters.convolve(u, m)) for m in masks[:4]])
        
        PM = PM + (1.0/2.0)*sum([np.multiply(np.exp(-np.square(ndimage.filters.convolve(u, m)/kappa)), ndimage.filters.convolve(u, m)) for m in masks[4:]])

        u = u + delta*PM

    return u
	
	
	
def getPts(imaniso, min_neuron=50, otsu_corr=0.85):
    from skimage import filters, morphology
    from skimage.feature import peak_local_max
    import math

    mask = imaniso < filters.threshold_otsu(imaniso)*otsu_corr
    mask = morphology.remove_small_objects(mask, min_size=min_neuron, connectivity=1, in_place=False)

    coordinates = peak_local_max(imaniso.max()-imaniso, min_distance=int(math.ceil(math.sqrt(min_neuron/3.14))))
    return coordinates[[mask[tuple(x)] for x in coordinates]]