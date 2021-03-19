#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from openni import openni2
import platform
import numpy as np
from cv2 import cv2
from time import sleep
from math import cos,sin,atan,sqrt,pow

# Initialize OpenNI
if platform.system() == "Windows":
    openni2.initialize("C:/Program Files/OpenNI2/Redist")  # Specify path for Redist
else:
    openni2.initialize()  # can also accept the path of the OpenNI redistribution

# Connect and open device
dev = openni2.Device.open_any()

# Create depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()

frame = depth_stream.read_frame()
frame_data = frame.get_buffer_as_uint16()
depth_array = np.asarray(frame_data).reshape((60, 80))

depth_correction = np.zeros((60, 80))
for y in range(0,60):
    for x in range(0,80):
        xc = (-79/2+x)
        yc = (59/2-y)*0.964773
        ac = (xc**2 + yc**2)**0.5
        rc = 52.41827
        R = (ac**2 + rc**2)**0.5
        depth_correction[y,x]=rc/R
        #alpha=atan(/)
        #r2 = (1-(cos(alpha)**2)-(sin(alpha)**2))
        #depth_correction[y,x]=sin(alpha)**2
        #depth_correction(y,x)=sqrt(1-cos(alpha)-sin(alpha))
        #depth_correction[y,x]=alpha

#print(depth_correction)


cv2.namedWindow("Filtered", cv2.WINDOW_NORMAL)
cv2.namedWindow("Raw In", cv2.WINDOW_NORMAL)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

if False:
    print ("MinTreshold: " + str(params.minThreshold) )
    print ("MaxTreshold: " + str(params.maxThreshold) )
    print ("filterByArea: " + str(params.filterByArea) )
    print ("minArea: " + str(params.minArea) )
    print ("maxArea: " + str(params.maxArea) )
    print ("filterByCircularity: " + str(params.filterByCircularity) )
    print ("minCircularity: " + str(params.minCircularity) )
    print ("maxCircularity: " + str(params.maxCircularity) )
    print ("filterByConvexity: " + str(params.filterByConvexity) )
    print ("minConvexity: " + str(params.minConvexity) )
    print ("maxConvexity: " + str(params.maxConvexity) )
    print ("filterByInertia: " + str(params.filterByInertia) )
    print ("minInertiaRatio: " + str(params.minInertiaRatio) )
    print ("maxInertiaRatio: " + str(params.maxInertiaRatio) )

    print ("minRepeatability: " + str(params.minRepeatability) )
    print ("minDistBetweenBlobs: " + str(params.minDistBetweenBlobs) )
    print ("thresholdStep: " + str(params.thresholdStep) )
    print ("blobColor: " + str(params.blobColor) )
    #print ("maxInertiaRatio: " + str(params.maxInertiaRatio) )

# Change thresholds
params.minThreshold = 1
params.maxThreshold = 200

# Filter by Area
params.filterByArea = True
params.minArea = 50
params.maxArea = 1000 # 600

# Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.1
#params.maxCircularity = 

# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87
#params.maxConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
#params.maxInertiaRatio = 0.01

params.minRepeatability = 15
params.minDistBetweenBlobs = 1 #1
params.thresholdStep = 1 #10
params.blobColor = 255

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(parameters=params)

print ("---------------------")

print ("MinTreshold: " + str(params.minThreshold) )
print ("MaxTreshold: " + str(params.maxThreshold) )
print ("filterByArea: " + str(params.filterByArea) )
print ("minArea: " + str(params.minArea) )
print ("maxArea: " + str(params.maxArea) )
print ("filterByCircularity: " + str(params.filterByCircularity) )
print ("minCircularity: " + str(params.minCircularity) )
print ("maxCircularity: " + str(params.maxCircularity) )
print ("filterByConvexity: " + str(params.filterByConvexity) )
print ("minConvexity: " + str(params.minConvexity) )
print ("maxConvexity: " + str(params.maxConvexity) )
print ("filterByInertia: " + str(params.filterByInertia) )
print ("minInertiaRatio: " + str(params.minInertiaRatio) )
print ("maxInertiaRatio: " + str(params.maxInertiaRatio) )

print ("minRepeatability: " + str(params.minRepeatability) )
print ("minDistBetweenBlobs: " + str(params.minDistBetweenBlobs) )
print ("thresholdStep: " + str(params.thresholdStep) )
print ("blobColor: " + str(params.blobColor) )
#print ("maxInertiaRatio: " + str(params.maxInertiaRatio) )

done = False
lastKeypointsCount = 0
reload = False

while not done and cv2.getWindowProperty("Filtered", cv2.WND_PROP_FULLSCREEN) != -1:

    key = cv2.waitKey(1)
    if key != -1: 
        if key == 27: # esc quits
            print ("Ending program... " )
            done = True
        elif key == 65: # A Increases minArea
            params.minArea = params.minArea + 1
            print ("+minArea: " + str(params.minArea) )
            reload = True
        elif key == 97: # a decreases minArea
            params.minArea = params.minArea - 1
            print ("-minArea: " + str(params.minArea) )
            reload = True
        elif key == 66: # B Increases maxArea
            params.maxArea = params.maxArea + 1
            print ("+maxArea: " + str(params.maxArea) )
            reload = True
        elif key == 98: # b decreases maxArea
            params.maxArea = params.maxArea - 1
            print ("-maxArea: " + str(params.maxArea) )
            reload = True
        elif key == 67: # C Increases blobColor
            params.blobColor = params.blobColor + 1
            print ("+blobColor: " + str(params.blobColor) )
            reload = True
        elif key == 99: # c decreases blobColor
            params.blobColor = params.blobColor - 1
            print ("-blobColor: " + str(params.blobColor) )
            reload = True
        elif key == 68: # D Increases minDistBetweenBlobs
            params.minDistBetweenBlobs = params.minDistBetweenBlobs + 1
            print ("+minDistBetweenBlobs: " + str(params.minDistBetweenBlobs) )
            reload = True
        elif key == 100: # d decreases minDistBetweenBlobs
            params.minDistBetweenBlobs = params.minDistBetweenBlobs - 1
            print ("-minDistBetweenBlobs: " + str(params.minDistBetweenBlobs) )
            reload = True
        elif key == 77: # M Increases minThreshold
            params.minThreshold = params.minThreshold + 1
            print ("+minThreshold: " + str(params.minThreshold) )
            reload = True
        elif key == 109: # m decreases minThreshold
            params.minThreshold = params.minThreshold - 1
            print ("-minThreshold: " + str(params.minThreshold) )
            reload = True
        elif key == 78: # N Increases maxThreshold
            params.maxThreshold = params.maxThreshold + 1
            print ("+maxThreshold: " + str(params.maxThreshold) )
            reload = True
        elif key == 110: # n decreases maxThreshold
            params.maxThreshold = params.maxThreshold - 1
            print ("-maxThreshold: " + str(params.maxThreshold) )
            reload = True
        elif key == 82: # R Increases minRepeatability
            params.minRepeatability = params.minRepeatability + 1
            print ("+minRepeatability: " + str(params.minRepeatability) )
            reload = True
        elif key == 114: # r decreases minRepeatability
            if params.minRepeatability > 1:
                params.minRepeatability = params.minRepeatability - 1
            print ("-minRepeatability: " + str(params.minRepeatability) )
            reload = True
        elif key == 84: # T Increases thresholdStep
            params.thresholdStep = params.thresholdStep + 1
            print ("+thresholdStep: " + str(params.thresholdStep) )
            reload = True
        elif key == 116: # t decreases thresholdStep
            if params.thresholdStep > 1:
                params.thresholdStep = params.thresholdStep - 1
            print ("-thresholdStep: " + str(params.thresholdStep) )
            reload = True
        else:
            print("keyPressed: " + str(key))
        if  reload == True: # reload parameters
            del detector
            detector = cv2.SimpleBlobDetector_create(parameters=params)
            #print ("-SimpleBlobDetector_Updated ")
            reload = False


   # sleep(0.01)
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_array_0 = np.asarray(frame_data).reshape((60, 80)) #* depth_correction
    
    depth_array = depth_array * 4 
    depth_array = depth_array + depth_array_0 
    depth_array = depth_array / 5
    depth_array = depth_array.astype(np.uint16)
    #frame_mean_2 = frame_mean.astype(np.uint16)

    # Trimming depth_array
    max_distance = 500 #1900 # 7000
    min_distance = 420 # 0
    out_of_range = depth_array > max_distance
    too_close_range = depth_array < min_distance
    depth_array[out_of_range] = max_distance
    depth_array[too_close_range] = min_distance

    # Scaling depth array
    #depth_scale_factor = 255.0 / (max_distance - min_distance)
    depth_scale_factor = 20.0 / (max_distance - min_distance)
    depth_scale_offset = -(min_distance * depth_scale_factor) + 50
    # depth_scale_factor = 255.0 / (np.amax(depth_array) - np.amin(depth_array))
    # depth_scale_offset = -(np.amax(depth_array) * depth_scale_factor)
    depth_array_norm = depth_array * depth_scale_factor + depth_scale_offset

    rgb_frame = cv2.applyColorMap(depth_array_norm.astype(np.uint8), cv2.COLORMAP_JET)


    # Replacing invalid pixel by black color
    #rgb_frame[np.where(depth_array == min_distance)] = [0, 0, 0]
    rgb_frame[np.where(depth_array == min_distance)] = [50, 50, 50]
    rgb_frame[np.where(depth_array >= 470) ] = [0, 250, 0]
    rgb_frame[np.where(depth_array == max_distance)] = [100, 100, 100]


    depth_array_in = depth_array_0
    out_of_range = depth_array_in > max_distance
    too_close_range = depth_array_in < min_distance
    depth_array_in[out_of_range] = max_distance
    depth_array_in[too_close_range] = min_distance
    depth_array_norm_in = depth_array_in * depth_scale_factor + depth_scale_offset

    rgb_frame_result = cv2.applyColorMap(depth_array_norm_in.astype(np.uint8), cv2.COLORMAP_JET)

    rgb_frame_result[np.where(depth_array_in == min_distance)] = [50, 50, 50]
    rgb_frame_result[np.where(depth_array_in == max_distance)] = [100, 100, 100]


    # Detect blobs.
    #keypoints = detector.detect(rgb_frame_result)
    keypoints = detector.detect(rgb_frame)
    #print(len(keypoints))
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    
    #im_with_keypoints = cv2.drawKeypoints(rgb_frame_result, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.drawKeypoints(rgb_frame, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    #cv2.imshow("Keypoints", im_with_keypoints)

    # Display image
   # rgb_frame = cv2.resize(rgb_frame, (800, 600), interpolation=cv2.INTER_AREA)
    #rgb_frame = cv2.resize(rgb_frame, (800, 600), interpolation=cv2.INTER_CUBIC)
    #rgb_frame = cv2.resize(rgb_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
   # rgb_frame_result = cv2.resize(rgb_frame_result, (800, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow("Filtered", rgb_frame)
    #cv2.imshow("Raw In", rgb_frame_result)
    cv2.imshow("Raw In", rgb_frame_result)

    if len(keypoints) > 0 :
        im_with_keypoints = cv2.resize(im_with_keypoints, (400, 300), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Keypoints", im_with_keypoints)
        if len(keypoints) != lastKeypointsCount:
            lastKeypointsCount = len(keypoints)
            #print ("Keypoints: " + str(lastKeypointsCount) )

            #for i in range(len(keypoints)):
            #    x = int(keypoints[i].pt[0])
            #    y = int(keypoints[i].pt[1])
            #    print (" x="+ str(x) + ", y=" + str(y))


depth_stream.stop()
openni2.unload()