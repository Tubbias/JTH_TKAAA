import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='YOLO training script')
    parser.add_argument('--image1-path', type=str, default='data/1_intro_da_source.jpg', help='Path to the first image')
    parser.add_argument('--image2-path', type=str, default='data/1_intro_da_target.png', help='Path to the second image')
    parser.add_argument('--method', type=str, required=False, default='SIFT', help="Feature detection method: 'ORB' or 'SIFT'")
    args = parser.parse_args()

    # Load images
    img1 = cv2.imread(args.image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the detector
    if args.method == 'ORB':
        detector = cv2.ORB_create()
    elif args.method == 'SIFT':
        detector = cv2.SIFT_create()
    else:
        raise ValueError("Unsupported method. Choose from 'ORB', or 'SIFT'.")

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if args.method == 'ORB' else cv2.NORM_L2, crossCheck=True)

    cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Warped Image', cv2.WINDOW_NORMAL)

    running = True
    rot_angle = 0
    rot_angle_changed = False
    while running:
        if rot_angle_changed:
            center = (img2.shape[1] // 2, img2.shape[0] // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            img2_process = cv2.warpAffine(img2, rot_matrix, (img2.shape[1], img2.shape[0]))
            rot_angle_changed = False
        else:
            img2_process = img2

        # Find the keypoints and descriptors with the chosen method
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2_process, None)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 30 matches
        img_matches = cv2.drawMatches(img1, kp1, img2_process, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Optional: Find homography and warp image
        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            img2_warped = cv2.warpPerspective(img2_process, H, (img1.shape[1], img1.shape[0]))

            cv2.imshow('Warped Image', img2_warped)

        # Draw information text
        info_text = f"Rotation Angle: {rot_angle} degrees | Press 'w'/'e' to rotate, 'q' to quit"
        cv2.putText(img_matches, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the matches
        cv2.imshow('Feature Matches', img_matches)
        key = cv2.waitKey(0)

        if key == ord('q'):
            running = False
        elif key == ord('w'):
            rot_angle += 10
            rot_angle_changed = True
        elif key == ord('e'):
            rot_angle -= 10
            rot_angle_changed = True

        rot_angle = rot_angle % 360

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
