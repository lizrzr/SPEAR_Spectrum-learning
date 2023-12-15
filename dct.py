import numpy
import torch
import cv2

#spectral loss extraction
def tensor_dct2D(ten, dim):
    num = ten.detach().cpu().numpy()
    if dim == 1:
        num1 = num[0, 0, :, :]
        dct = cv2.dct(num1)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0

        idct = cv2.idct(dct)
        result = torch.FloatTensor(idct)
        return result
    if dim == 2:
        num1 = num[0, 0, :, :]
        dct = cv2.dct(num1)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0
        idct1 = cv2.idct(dct)

        num2 = num[1, 0, :, :]
        dct = cv2.dct(num2)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0
        idct2 = cv2.idct(dct)
        result = numpy.array([idct1, idct2])
        result = torch.FloatTensor(result)
        return result
    if dim == 4:
        num1 = num[0, 0, :, :]
        dct = cv2.dct(num1)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0
        idct1 = cv2.idct(dct)

        num2 = num[1, 0, :, :]
        dct = cv2.dct(num2)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0
        idct2 = cv2.idct(dct)

        num3 = num[2, 0, :, :]
        dct = cv2.dct(num3)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0
        idct3 = cv2.idct(dct)

        num4 = num[3, 0, :, :]
        dct = cv2.dct(num4)
        for i in range(100):
            for j in range(100):
                dct[i, j] = 0
        idct4 = cv2.idct(dct)

        result = numpy.array([idct1, idct2, idct3, idct4])
        result = torch.FloatTensor(result)
        return result
