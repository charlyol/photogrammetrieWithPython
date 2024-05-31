import cv2
import numpy as np
import open3d as o3d
import glob
import pickle

path_to_images = 'data/frames_shutil/*.jpg'


def load_images(img_paths):

    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images


def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2


def compute_3d_points(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)

    return points_3d[:, 0, :]


def visualize_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])


def main():
    img_paths = glob.glob('../data/frames_shutil/*.jpg')  # Chemin du dossier contenant les images
    num_images = len(img_paths)

    # Charger les paramètres calibrés depuis le fichier texte
    with open('calibration.pkl', 'rb') as f:
        calibration_data = pickle.load(f)
        mtx = calibration_data[0]
    print (num_images)
    for i in range(num_images-1):
        img1, img2 = load_images([img_paths[i], img_paths[i + 1]])
        pts1, pts2 = detect_and_match_features(img1, img2)
        points_3d = compute_3d_points(pts1, pts2, mtx)
        visualize_point_cloud(points_3d)


if __name__ == "__main__":
    main()
