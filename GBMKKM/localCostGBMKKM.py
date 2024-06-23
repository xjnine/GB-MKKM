from functions.sumKbeta import sumKbeta
from GBMKKM.mylocalkernelkmeans import mylocalkernelkmeans


def localCostGBMKKM(KH, StepSigma, DirSigma, A, Sigma, numclass):
    global nbcall
    nbcall = 0
    nbcall = nbcall + 1

    Sigma = Sigma + StepSigma * DirSigma
    Kmatrix = sumKbeta(KH, (Sigma * Sigma))
    [Hstar, cost] = mylocalkernelkmeans(Kmatrix, A, numclass)
    return cost, Hstar
