import numpy as np
import cv2
import glob
import sys
import copy
import math

cap = cv2.VideoCapture(0)
ret = cap.set(3,640)
ret = cap.set(4,480)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

gridW = 8
gridH = 6

#sift = cv2.SIFT()

def draw2(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def reprojectionError(rvecs,tvecs,mtx,dist):
    mean_error = 0
    tot_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs, tvecs, mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    mean_error = tot_error/len(objpoints)
    print "total error: ", tot_error, ' mean error: ', mean_error

def getCornersChessBoard(gray):
    ret, corners = cv2.findChessboardCorners(gray, (gridW,gridH),None)
    corners2 = copy.deepcopy(corners)

    return (ret, corners2)

def getCornersSIFT(image, template):
    # find the keypoints and descriptors with SIFT
    kp_image, image_des = sift.detectAndCompute(image,None)
    kp_template, template_des = sift.detectAndCompute(template,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(image_des,template_des,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    ret = False

    if len(good) > 10:
        ret = True

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    return (ret, dst_pts)



def exit():
    cap.release()
    cv2.destroyAllWindows() 
    print 'Bye bye!'
    sys.exit(0)

def main():

    data = np.load('cameraParams.npz')
    mtx = data['intrinsic']
    dist = data['distortion']


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)
    objp = np.zeros((gridW*gridH,3), np.float32)
    
    objp[:,:2] = np.mgrid[0:gridW,0:gridH].T.reshape(-1,2)

    #axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

    #template = cv2.imread('template.jpg',0)

    key = -1
    while key!=27:
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners

        ret, corners2 = getCornersChessBoard(gray)

        key = cv2.waitKey(100)
        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners2,(13,13),(-1,-1),criteria)

            imgpoints.append(corners2)

            # Find the rotation and translation vectors.
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            reprojectionError(rvecs,tvecs,mtx,dist)

            # project 3D points to image plane

            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)



            img = draw(img,corners2,imgpts)

            print 'CamPos: ',tvecs, ' CamOr: ', (rvecs*180/math.pi)

        cv2.imshow('img',img)

    exit()

    




if  __name__ =='__main__':main()