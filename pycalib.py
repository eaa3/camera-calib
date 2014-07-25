import numpy as np
import cv2
import glob
import sys
import copy



# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


#images = glob.glob('*.jpg')
cap = cv2.VideoCapture(0)
ret = cap.set(3,640)
ret = cap.set(4,480)

def exit():
   cap.release()
   cv2.destroyAllWindows() 
   print 'Bye bye!'
   sys.exit(0)


#### MAIN ####
def main():

    ngood_images = 0
    key = -1


    while ngood_images < 10:
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if ret == True:
            cv2.imshow('img',img)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        corners2 = copy.deepcopy(corners)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners2,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(2000)
            ngood_images = ngood_images + 1

        
        key = cv2.waitKey(1)
        if key == 27:
            exit()

    
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    #np.savez('cameraParams',intrinsic=mtx,distortion=dist)
    mean_error = 0
    tot_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    mean_error = tot_error/len(objpoints)
    print "total error: ", tot_error, ' mean error: ', mean_error

    while key != 27:
        ret, img = cap.read()

        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('undistorted',dst)
        cv2.imshow('img',img)
        key = cv2.waitKey(1)

    

    exit()



if  __name__ =='__main__':main()



