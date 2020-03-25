from cv2 import cv2
import numpy as np
import zzimgtool

img = cv2.imread('face/hair.png')
mask = cv2.imread('face/hairmask.png')
colormap = cv2.imread('face/haircolor.png')
colormap[:,:,:] = 255
# colormap[:,:,1] = 255

A = img.astype(np.float)
B = colormap.astype(np.float)
C = np.zeros_like(A, dtype=np.float)
whr_0_127 = np.where(B <= 128)
whr_128_255 = np.where(B > 128)

C[whr_128_255] = A[whr_128_255] + (2 * B[whr_128_255] - 255) * (np.sqrt(A[whr_128_255] / 255) * 255 - A[whr_128_255]) / 255

C[whr_0_127] = A[whr_0_127] + (2 * B[whr_0_127] - 255) * (A[whr_0_127] - A[whr_0_127] * A[whr_0_127] / 255) / 255

C = C.astype(np.uint8)

res = zzimgtool.AlphaBlending(fgImg=C, bgImg=A, alphaMask=mask)
cv2.imshow("winname", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

for ab in zip(AB):
    for abrow in ab:
        for abcol in abrow:
            for abpix in abcol:
                
                '''