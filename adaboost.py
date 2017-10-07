from numpy import *


def loadSimpData():
    datMat = matrix([[1, 2.1],
                     [2, 1.1],
                     [1.3, 1],
                     [1, 1],
                     [2, 1]])
    classLabels = [1, 1, -1, -1, 1]
    return datMat, classLabels
