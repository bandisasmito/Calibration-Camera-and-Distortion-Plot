# remove_distortion.py
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

def load_calibration_results(calibration_file):
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration results file '{calibration_file}' not found.")
        return None, None, None

    with np.load(calibration_file) as data:
        rep_error = data['rep_error']
        cam_matrix = data['cam_matrix']
        dist_coeff = data['dist_coeff']

    return rep_error, cam_matrix, dist_coeff

def display_calibration_results(calibration_file):
    rep_error, cam_matrix, dist_coeff = load_calibration_results(calibration_file)

    if rep_error is None or cam_matrix is None or dist_coeff is None:
        print("Error: Unable to load calibration results.")
        return

    print("\nCamera Matrix:")
    print(cam_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeff)
    print("\nReprojection Error (pixels):", rep_error)

def remove_distortion(calibration_file):
    rep_error, cam_matrix, dist_coeff = load_calibration_results(calibration_file)

    if rep_error is None or cam_matrix is None or dist_coeff is None:
        print("Error: Unable to load calibration results.")
        return

    root = os.getcwd()
    img_path = os.path.join(root, 'D:/RDD/calib/1/dist/aa.jpg')
    img = cv.imread(img_path)

    height, width = img.shape[:2]
    cam_matrix_new, roi = cv.getOptimalNewCameraMatrix(cam_matrix, dist_coeff, (width, height), 1, (width, height))
    img_undist = cv.undistort(img, cam_matrix, dist_coeff, None, cam_matrix_new)

    # Display calibration results
    display_calibration_results(calibration_file)

    # Draw Line to See Distortion Change
    cv.line(img, (1769, 103), (1780, 922), (255, 255, 255), 2)
    cv.line(img_undist, (1769, 103), (1780, 922), (255, 255, 255), 2)

    plt.figure() 
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_undist)
    plt.show()

def run_remove_distortion():
    calibration_file = 'D:/RDD/calib/1/result/calibration_results.npz'  # Sesuaikan dengan lokasi file kalibrasi
    remove_distortion(calibration_file)

if __name__ == '__main__':
    run_remove_distortion()
