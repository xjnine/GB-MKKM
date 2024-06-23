import numpy as np
from functions.sumKbeta import sumKbeta
from GBMKKM.mylocalkernelkmeans import mylocalkernelkmeans
from GBMKKM.localGBMKKMGrad import localGBMKKMGrad
from GBMKKM.localGBMKKMupdate import localGBMKKMupdate


def localizedGBMKKM(KH, numclass, A, options):
    numker = KH.shape[0]  # 核数
    sigma = np.ones((numker, 1)) / numker
    if 'goldensearch_deltmax' not in options:
        options['goldensearch_deltmax'] = 5e-2
    if 'goldensearchmax' not in options:
        options['goldensearchmax'] = 1e-8
    if 'firstbasevariable' not in options:
        options['firstbasevariable'] = 'first'
    nloop = 0
    loop = 1
    obj = []
    goldensearch_deltmaxinit = options['goldensearch_deltmax']

    # Initializing Kernel K-means
    Kmatrix = sumKbeta(KH, sigma ** 2)
    Hstar, obj1 = mylocalkernelkmeans(Kmatrix, A, numclass)
    obj.append(obj1)
    grad = localGBMKKMGrad(KH, A, Hstar, sigma)
    sigmaold = sigma
    while loop:
        nloop = nloop + 1
        # Update weigths Sigma
        sigma, Hstar, objv = localGBMKKMupdate(KH, sigmaold, grad, A, obj[nloop-1], numclass, options)
        obj.append(objv)
        # Enhance accuracy of line search if necessary
        if np.max(np.abs(
            sigma - sigmaold)) < options['numericalprecision'] and options['goldensearch_deltmax'] > options['goldensearchmax']:
            options['goldensearch_deltmax'] = options['goldensearch_deltmax'] / 10
        elif options['goldensearch_deltmax'] != goldensearch_deltmaxinit:
            options['goldensearch_deltmax'] = options['goldensearch_deltmax'] / 10
        grad = localGBMKKMGrad(KH, A, Hstar, sigma)
        # check variation of Sigma conditions
        if np.max(np.abs(sigma - sigmaold)) < options['seuildiffsigma']:
            loop = 0
            print("variation convergence criteria reached")
        sigmaold = sigma

    return Hstar, sigma, obj
