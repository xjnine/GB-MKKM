import numpy as np
from GBMKKM.localCostGBMKKM import localCostGBMKKM


def localGBMKKMupdate(KH, Sigma, GradNew, A, CostNew, numclass, option):
    gold = (np.sqrt(5) + 1) / 2
    SigmaInit = Sigma
    SigmaNew = SigmaInit
    NormGrad = np.dot(GradNew.T, GradNew)
    GradNew = GradNew / np.sqrt(NormGrad)
    CostOld = CostNew

    # Compute reduced Gradient and descent direction
    if option['firstbasevariable'] == 'first':
        val, coord = np.max(SigmaNew), np.argmax(SigmaNew)
    elif option['firstbasevariable'] == 'random':
        val, coord = np.max(SigmaNew), np.argmax(SigmaNew)
        coord = np.where(SigmaNew == val)[0]
        indperm = np.random.permutation(len(coord))  # # 生成 coord 长度的随机排列
        # 从随机排列中取出第一个元素，并将其赋给 coord
        coord = coord[indperm[0]]
    elif option['firstbasevariable'] == 'fullrandom':
        indzero = np.where(SigmaNew != 0)[0]
        if len(indzero) > 0:
            mini, coord = np.min(GradNew[indzero]), np.argmin(GradNew[indzero])
            coord = indzero[coord]
        else:
            val, coord = np.max(SigmaNew), np.argmax(SigmaNew)

    GradNew = GradNew - GradNew[coord]
    desc = - GradNew * ((SigmaNew > 0) | (GradNew < 0))
    desc[coord] = -np.sum(desc)

    # Compute optimal stepsize
    stepmin = 0
    costmin = CostOld
    costmax = 0
    ind = np.where(desc < 0)[0]
    stepmax = np.min(-(SigmaNew[ind]) / desc[ind])
    deltmax = stepmax
    if stepmax is None or stepmax == 0:
        Sigma = SigmaNew
        return
    if stepmax > 0.1:
        stepmax = 0.1

    # Projected gradient
    while costmax < costmin:
        costmax, Hstar = localCostGBMKKM(KH, stepmax, desc, A, SigmaNew, numclass)
        if costmax < costmin:
            costmin = costmax
            SigmaNew = SigmaNew + stepmax * desc
            SigmaNew[np.abs(SigmaNew) < option['numericalprecision']] = 0
            SigmaNew = SigmaNew / np.sum(SigmaNew)
            desc = desc * ((SigmaNew > option['numericalprecision']) | (desc > 0))
            desc[coord] = -np.sum(desc[np.concatenate([np.arange(0, coord), np.arange(coord + 1, len(desc))])])
            ind = np.where(desc < 0)[0]
            if len(ind) > 0:
                stepmax = np.min(-SigmaNew[ind] / desc[ind])
                deltmax = stepmax
                costmax = 0
            else:
                stepmax = 0
                deltmax = 0
    Step = [stepmin, stepmax]
    Cost = [costmin, costmax]
    coord = np.argmin(Cost)
    val = Cost[coord]
    eps = np.finfo(float).eps
    # 假设 stepmin, stepmax, deltmax, option 是已知的值

    while (stepmax - stepmin) > option['goldensearch_deltmax'] * np.abs(deltmax) and stepmax > eps:
        # 假设 gold 是一个已知的值
        stepmedr = stepmin + (stepmax - stepmin) / gold
        stepmedl = stepmin + (stepmedr - stepmin) / gold
        # 计算右侧中间步长的代价和特征向量
        costmedr, Hstarr = localCostGBMKKM(KH, stepmedr, desc, A, SigmaNew, numclass)

        # 计算左侧中间步长的代价和特征向量
        costmedl, Hstarl = localCostGBMKKM(KH, stepmedl, desc, A, SigmaNew, numclass)
        # 更新 Step 和 Cost 数组
        Step = [stepmin, stepmedl, stepmedr, stepmax]
        Cost = [costmin, costmedl, costmedr, costmax]
        coord = np.argmin(Cost)
        val = Cost[coord]
        if coord == 0:
            stepmax = stepmedl
            costmax = costmedl
            Hstar = Hstarl
        elif coord == 1:
            stepmax = stepmedr
            costmax = costmedr
            Hstar = Hstarr
        elif coord == 2:
            stepmin = stepmedl
            costmin = costmedl
            Hstar = Hstarl
        elif coord == 3:
            stepmin = stepmedr
            costmin = costmedr
            Hstar = Hstarr

    #  Final Updates
    CostNew = Cost[coord]
    step = Step[coord]

    # Sigma update
    if CostNew < CostOld:
        SigmaNew = SigmaNew + step * desc
    Sigma = SigmaNew
    return Sigma, Hstar, CostNew
    print("@@")
