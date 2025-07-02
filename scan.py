from pyimagesearch.perspective import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", 
    "--image", 
    required=True, 
    help="Path to image to be scanned"
)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

print("#1. Edge Detection")
cv2.imshow("Image", image)  # original img
cv2.imshow("Edged", edged)  # edge detected img
cv2.waitKey(0)
cv2.destroyAllWindows()

print("#2. Find Contours")
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
screenCnt = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    print("No contour detected")
    exit(0)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)  # outline of the document
cv2.waitKey(0)
cv2.destroyAllWindows()

print("#3. Apply Perspective Transform")
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
thresh_loc = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > thresh_loc).astype("uint8") * 255

cv2.imshow("Original", imutils.resize(orig, height = 650))  # original image
cv2.imshow("Scanned", imutils.resize(warped, height = 650))  # scanned image
cv2.waitKey(0)