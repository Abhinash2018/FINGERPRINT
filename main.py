import cv2
import os

sample = cv2.imread(
    r"C:\Users\abhig\fingerprint\SOCOFing\Altered\Altered-Hard\1__M_Right_middle_finger_Obl.BMP",
    cv2.IMREAD_GRAYSCALE
)


best_score = 0
filename = None
image = None
kp1 = kp2 = mp = None

sift = cv2.SIFT_create()


for idx, file in enumerate(os.listdir("SOCOFing/Real")[:1000]):
    if idx % 10 == 0:
        print(idx)

    fingerimage = cv2.imread("SOCOFing/Real/" + file, cv2.IMREAD_GRAYSCALE)
    if fingerimage is None:
        continue

    keypoint_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoint_2, descriptors_2 = sift.detectAndCompute(fingerimage, None)

    if descriptors_1 is None or descriptors_2 is None:
        continue

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=10),
        dict(checks=50)
    )

    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    match_point = []
    for p, q in matches:
        if p.distance < 0.75 * q.distance:
            match_point.append(p)

    keypoints = min(len(keypoint_1), len(keypoint_2))
    score = len(match_point) / keypoints * 100

    if score > best_score:
        best_score = score
        filename = file
        image = fingerimage
        kp1, kp2, mp = keypoint_1, keypoint_2, match_point

print("Best Match:", filename)
print("Best Score:", best_score)

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=2, fy=2)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
