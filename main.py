import cv2
import os

path = r"C:\Users\abhig\fingerprint\SOCOFing\Altered\Altered-Hard\1__M_Left_index_finger_CR.BMP"

sample = cv2.imread(path)

if sample is None:
    print("Image not found. Check path.")
    exit()

sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

cv2.imshow("Sample", sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
