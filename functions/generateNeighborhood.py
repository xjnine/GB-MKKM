import numpy as np


# kcenter - center a kernel matrix
def generateNeighborhood(avgKer, numSel, direction='descend'):
    num = avgKer.shape[0]
    avgKer0 = avgKer - 1e8 * np.eye(num)

    # Sort the matrix along the specified direction
    if direction == 'descend':
        sorted_indices = np.argsort(-avgKer0, axis=1)
    else:
        sorted_indices = np.argsort(avgKer0, axis=1)

    # Extract the first 'tau' indices from the sorted matrix
    indx_0 = sorted_indices[:, :numSel]

    return indx_0



