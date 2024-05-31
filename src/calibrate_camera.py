import numpy as np
import cv2 as cv
import glob
import pickle
from tnoest_imshow import frameSize
from PIL import Image
import matplotlib.pyplot as plt
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8,5)
frameSize = frameSize('../data/little/IMG_9635.jpg')
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 21, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 21
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('../data/little/IMG_9635.jpg')

print(f"Nombre d'images trouvées : {len(images)}")

for image in images:

    img = cv.imread(image)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Trouver les coins de l'échiquier
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # Si trouvé, ajouter des points d'objet, des points d'image (après les avoir affinés)
    if ret:
        print(f"Corners d'échiquier trouvés dans l'image {image}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Dessiner et afficher les coins
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        imgplot = plt.imshow(img)
        cv.imshow('img', img)
        # cv.waitKey(1000)
    else:
        print(f"Corners d'échiquier non trouvés dans l'image {image}")

cv.destroyAllWindows()

############## CALIBRATION #######################################################

if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
else:
    print("No chessboard corners were found in any image.")
    exit(1)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)

pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
pickle.dump(dist, open("dist.pkl", "wb"))

############## UNDISTORTION #####################################################
img = cv.imread('../data/little/frame_0043.jpg')

h = (frameSize[0])
w = (frameSize[1])
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('caliResult1.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('caliResult2.png', dst)

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))