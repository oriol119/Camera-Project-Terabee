#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from openni import openni2
import platform
import numpy as np
from cv2 import cv2
from time import sleep
from math import cos,sin,atan,sqrt,pow
import math
import threading
import time
global frame_fons
frame_fons = 0

global contador 
contador = 0



 






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

frame_threaD = depth_stream.read_frame()
frame_threaD_ok = frame_threaD.get_buffer_as_uint16()
frame_threaD_resize = np.asarray(frame_threaD_ok).reshape((60, 80)) #* depth_correction




valor_complert = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
class thread(threading.Thread): 
     
    
    def __init__(self, thread_name, thread_ID):  
        threading.Thread.__init__(self)  
        self.thread_name = thread_name  
        self.thread_ID = thread_ID  
  
        # helper function to execute the threads 
    def run(self):
        global contador
        #a= np.zeros(4800)
        #ab = np.asarray(a).reshape((60, 80))
        contador = 0
        while(True):
            start_time = time.time()
            interval = 0.1
            

            for i in valor_complert:
                


                time.sleep(abs(start_time + i*interval - time.time()))

                frame_thread = depth_stream.read_frame()
                frame_thread_ok = frame_thread.get_buffer_as_uint16()
                frame_thread_resize = np.asarray(frame_thread_ok).reshape((60, 80)) #* depth_correction
            

             
                if i == 0:
                    global aTotal
                    
                    global frame_fons
                    #a0 = frame_thread_resize
                    
                    #print("1r:")
                    #print(a0[30][30])
                    aTotal = frame_thread_resize
                    print("0")
                elif i == 1:
                    a1 = frame_thread_resize
                    #print("2n:")
                    #print(a1[30][30])
                    aTotal = (aTotal + a1)/2
                    print("1")

                    
                elif i == 2:
                    a2 = frame_thread_resize
                    #print("3r:")
                    #print(a2[30][30])
                    aTotal = (aTotal + a2)/2
                    print("2")

                elif i == 3:
                    a3 = frame_thread_resize
                    #print("3r:")
                    #print(a3[30][30])
                    aTotal = (aTotal + a3)/2
                    print("3")

                elif i == 4:
                    a4 = frame_thread_resize
                    #print("3r:")
                    #print(a4[30][30])
                    aTotal = (aTotal + a4)/2
                    print("4")

                elif i == 5:
                    a5 = frame_thread_resize
                    #print("3r:")
                    #print(a5[30][30])
                    aTotal = (aTotal + a5)/2
                    print("5")

                elif i == 6:
                    a6 = frame_thread_resize
                    #print("3r:")
                    #print(a6[30][30])
                    aTotal = (aTotal + a6)/2
                    print("6")

                elif i == 7:
                    a7 = frame_thread_resize
                    #print("3r:")
                    #print(a7[30][30])
                    aTotal = (aTotal + a7)/2
                    print("7")

                elif i == 8:
                    a8 = frame_thread_resize
                    #print("3r:")
                    #print(a8[30][30])
                    aTotal = (aTotal + a8)/2
                    print("8")

                elif i == 9:
                    a9 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a9)/2
                    print("9")

                elif i == 10:
                    a10 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a10)/2
                    print("10")

                elif i == 11:
                    a11 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a11)/2
                    print("11")

                elif i == 12:
                    a12 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a12)/2
                    print("12")

                elif i == 13:
                    a13 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a13)/2
                    print("13")

                elif i == 14:
                    a14 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a14)/2
                    print("14")

                elif i == 15:
                    a15 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a15)/2
                    print("15")

                elif i == 16:
                    a16 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a16)/2
                    print("16")

                elif i == 17:

                    a17 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a17)/2
                    print("17")

                elif i == 18:

                    a18 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a18)/2
                    print("18")

                elif i == 19:

                    a19 = frame_thread_resize
                    #print("3r:")
                    #print(a9[30][30])
                    aTotal = (aTotal + a19)/2
                    print("19")



                    if contador == 0:
                        frame_fons = aTotal
                        
                        contador += 1
                 
                       



thread1 = thread("GFG", 1000)
thread1.start()




flag = 0


frame = depth_stream.read_frame()
frame_data3 = frame.get_buffer_as_uint16()





#print(len(frame_data))

depth_array = np.asarray(frame_data3).reshape((60, 80))

depth_correction = np.zeros((60, 80))
for y in range(0,60):
    for x in range(0,80):
        xc = (-79/2+x)
        yc = (59/2-y)*0.964773
        ac = (xc**2 + yc**2)**0.5
        rc = 52.41827
        R = (ac**2 + rc**2)**0.5
        depth_correction[y,x]=rc/R


cv2.namedWindow("Filtered", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Raw In", cv2.WINDOW_NORMAL)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()



if False:
    print ("MinTreshold: " + str(params.minThreshold))
    print ("MaxTreshold: " + str(params.maxThreshold))
    print ("filterByArea: " + str(params.filterByArea))
    print ("minArea: " + str(params.minArea))
    print ("maxArea: " + str(params.maxArea))
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
params.maxArea = 600 # 600

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.3
params.maxCircularity = 0.5

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
    mitja = cv2.applyColorMap(aTotal.astype(np.uint8), cv2.COLORMAP_JET)

    #mitja[np.where(aTotal > 557)] = [255,0,0] #blau


    p_ok = cv2.resize(mitja, (800, 500), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("FILTRE", p_ok)

  


    #mitja[np.where(aTotal > 557)] = [255, 20, 0]
    #mitja[np.where(aTotal < 535)] = [255, 20, 0]


    
    #correccio = cv2.applyColorMap(mitja.astype(np.uint8), cv2.COLORMAP_JET)
    #correccio_ok = cv2.resize(correccio, (800, 500), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("CORRECCIO", correccio_ok)


    if contador == 1:


        static_frame_color = cv2.applyColorMap(frame_fons.astype(np.uint8), cv2.COLORMAP_JET)
        

        #static_frame_color[np.where(frame_fons < 535)] = [0, 0, 0]
        #mitja[np.where(frame_fons <= 545) ] = [0, 250, 0]
        static_frame_color[np.where(frame_fons > 557)] = [255,0,0]

        static_frame_ok = cv2.resize(static_frame_color, (800, 500), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("STATIC BACKGROUND", static_frame_ok) 

        


        resta_1 = aTotal - frame_fons
        #resta_2 = correccio - static_frame_color

        resta1 = cv2.applyColorMap(resta_1.astype(np.uint8), cv2.COLORMAP_JET)
        #resta2 = cv2.applyColorMap(resta_2.astype(np.uint8), cv2.COLORMAP_JET)
        


        
        resta1[np.where(resta_1 < -4)] = [0,0,0]  #negre
        resta1[np.where(resta_1 < -5)] = [255,0,0]  #blau
        resta1[np.where(resta_1 < -6)] = [0,255,0]  #verd

        resta1[np.where(resta_1 > -4)] = [255,255,255] #blanc
    
     

        resta1_ok = cv2.resize(resta1, (800, 500), interpolation=cv2.INTER_LINEAR)
        #resta2_ok = cv2.resize(resta2, (800, 500), interpolation=cv2.INTER_LINEAR)



        cv2.imshow("RESTA", resta1_ok) 
        #cv2.imshow("RESTA2", resta2_ok) 







    #correccio2 = cv2.applyColorMap(mitja.astype(np.uint8), cv2.COLORMAP_JET)
    #correccio_ok2 = cv2.resize(correccio2, (800, 500), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("CORRECCIO2", correccio_ok2)

    #print("Total Definitiu:")
    #print(aTotal[30][30])      


    depth_scale_factor = 255.0 / (560 - 530)
    depth_scale_offset = -(530 * depth_scale_factor) + 0
    # depth_scale_factor = 255.0 / (np.amax(depth_array) - np.amin(depth_array))
    # depth_scale_offset = -(np.amax(depth_array) * depth_scale_factor)
    depth_array_norm = aTotal * depth_scale_factor + depth_scale_offset

    scaledd = cv2.applyColorMap(depth_array_norm.astype(np.uint8), cv2.COLORMAP_JET)



    key = cv2.waitKey(1)
    if key != -1: 
        if key == 27: # esc quits
            print ("Ending program... " )
            done = True
        elif key == 65: # A Increases minArea
            params.minArea = params.minArea + 1
            print ("+minArea: " + str(params.minArea))
            reload = True
        elif key == 97: # a decreases minArea
            params.minArea = params.minArea - 1
            print ("-minArea: " + str(params.minArea))
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

    
    provv2 = cv2.applyColorMap(depth_array_0.astype(np.uint8), cv2.COLORMAP_JET)
    provv3 = cv2.resize(provv2, (800, 500), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("vista original", provv3)
    
    depth_array = depth_array * 4 
    depth_array = depth_array + depth_array_0 
    depth_array = depth_array / 5
    depth_array = depth_array.astype(np.uint16)
    #frame_mean_2 = frame_mean.astype(np.uint16)

    # Trimming depth_array
    max_distance = 600 #1900 #7000
    min_distance = 540 # 0
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
    rgb_frame[np.where(depth_array == min_distance)] = [255, 255, 255]
    rgb_frame[np.where(depth_array <= 545) ] = [0, 250, 0]
    rgb_frame[np.where(depth_array == max_distance)] = [255, 255, 255]


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
    
    

    #bucle per trobar contorn de l'objecte i dibuixar-lo:
    #bucle per trobar el centre i les diagonals de l'objecte i dibuixar-lo:

    edged = cv2.Canny(rgb_frame, 170, 255)   #Determine edges of objects in an image

    gray_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,110,255,cv2.THRESH_BINARY)  
    (contours,_) = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours in an image
    
   
    #if flag == 0:
    for i in [0,1,2,3,4,5,6]:
        if i == 0:   

            captura_imatge = frame_data 
                    
        else:
            for y in range(len(captura_imatge)):

                captura_imatge[y] = frame_data[y] + captura_imatge[y]
    
                if i == 6:
                    captura_imatge[y] = int(captura_imatge[y] / 7) 
            



    depth_array_background = np.asarray(captura_imatge).reshape((60, 80)) #* depth_correction

    background_avarege = cv2.applyColorMap(depth_array_background.astype(np.uint8), cv2.COLORMAP_JET)

    background_avarege_ok = cv2.resize(background_avarege, (800, 500), interpolation=cv2.INTER_LINEAR)

    #cv2.imshow("background average", background_avarege_ok)
    



   











    
    #resta = frame_data 
    #aixo = captura_imatge


    #for y in range(len(resta)):
    #    resta[y] = resta[y] - aixo[y]
        
    #    if resta[y] == 0:
    #        resta[y] = 100

            #print("NO INTERESA")
    #    elif resta[y] > 0:
    #        resta[y] = 300
            #print("NO INTERESA")
    #    elif resta[y] < 0:
    #       resta[y] = 650
            #print("INTERESA")

    

         
    #resta_mida = np.asarray(resta).reshape((60, 80)) #* depth_correction
    #resta_color = cv2.applyColorMap(resta_mida.astype(np.uint8), cv2.COLORMAP_JET)
    #resta_ok = cv2.resize(resta_color, (800, 500), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("resta", resta_ok)
       








    
        

    #print(captura_imatge[4400])
    #depth_array_background[depth_array_background > 560 ] = 0
    #depth_array_background[depth_array_background < 560 ] = 2000
    #for x in range(len(depth_array_background)):
     #   if depth_array_background.all() == provv4.all():
      #      print("E")
       # else:
        #    print("O")

    
    

            


    

        


    #fons = captura_imatge

    #frame_capa2 = depth_stream.read_frame()
    #frame_capa2_data = frame_capa2.get_buffer_as_uint16()
    #capa2 = []
    #print(len(frame_capa2_data))
    #print(len(captura_imatge))

    flag += 1
    

    
    #for r in range(len(frame_capa2_data)):
     #   frame_capa2_data[r] -=  fons[r]
      #  print(frame_capa2_data[7])
   

    #vista_array = np.asarray(frame_capa2_data).reshape((60, 80)) #* depth_correction

    #resultat = vista_array - depth_array_background

    #nova_capa = cv2.applyColorMap(vista_array.astype(np.uint8), cv2.COLORMAP_JET)
    #nova_capa_ok = cv2.resize(nova_capa, (800, 500), interpolation=cv2.INTER_LINEAR)

    #cv2.imshow("nova capa", nova_capa_ok)




  

    
   

    keypoints = detector.detect(background_avarege)

    im_with_keypoints = cv2.drawKeypoints(background_avarege, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    
    for cnt in contours:

        blob = max(contours, key=lambda el: cv2.contourArea(el))
        M = cv2.moments(cnt)
    
        area = cv2.contourArea(cnt)

        #if area > 100:
            #print(area)

        cv2.drawContours(im_with_keypoints,[cnt],-1,(0,0,0),1)
        
        # calculate moments for each contour
        
        if M["m00"] != 0 :
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            #dibuixem el centre
            cv2.line(im_with_keypoints, center, center, (0,0,255), 1)

        rect = cv2.minAreaRect (cnt)
        angulo_objeto = rect[2]
        #print(angulo_objeto)
        box = cv2.boxPoints (rect)
        
        a = box[0]
        b = box[1]
        c = box[2]
        d = box[3]

        box = np.int0(box)
        

        #NOOO BORRAAAAR
        #NOOO BORRAAAAR

        #dibuixem el rectangle
        
        #cv2.drawContours(rgb_frame,[box],0,(0,0,255),1)


        #dibuixem diagonals
        
        #cv2.line(rgb_frame, (int(a[0]),int(a[1])), (int(c[0]),int(c[1])), (255,0,0), 1)
        #cv2.line(rgb_frame, (int(b[0]),int(b[1])), (int(d[0]),int(d[1])), (255,0,0), 1)
            

        #utilitzem els valors de HSV per a separar la barra de pa del reste de contingut en una màscara
        imgHSV = cv2.cvtColor(rgb_frame,cv2.COLOR_BGR2HSV)
        h_min = 100
        h_max = 255
        s_min = 100
        s_max = 255
        v_min = 100
        v_max = 255
        
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv2.inRange(imgHSV,lower,upper)
        
        mask = cv2.resize(mask, (800, 600), interpolation=cv2.INTER_AREA)
                
            
            #a = (moment['m20'] / moment['m00']) 
            #b = 2 * (moment['m11'] / moment['m00']) 
            #c = (moment['m02'] / moment['m00']) 
            #if a-c != 0:
             #   theta = 0.5 * np.arctan (b/(a-c))
              #  grados = math.degrees(theta)
               # if len(cnt) > 5:
                #    ellipse = cv2.fitEllipse(cnt)
                 #   im = cv2.ellipse(rgb_frame_result,ellipse,(0,255,0),1)

                    #print(grados)
    edged = cv2.resize(edged, (800, 500), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("Contornos", edged)




    edged = cv2.Canny(background_avarege, 170, 255)   #Determine edges of objects in an image

    gray_image = cv2.cvtColor(background_avarege, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,110,255,cv2.THRESH_BINARY)  
    (contours,_) = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours in an image


    edged = cv2.resize(edged, (800, 500), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("Contornos2", edged)


    gray_image_ok = cv2.resize(gray_image, (800, 500), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("GRIS", gray_image_ok)

    

    #cv2.imshow("Mascara", mask)
    #cv2.imshow("Filtered", rgb_frame)
    #cv2.imshow("Raw In", rgb_frame_result)
    #cv2.imshow("Raw In", rgb_frame_result)

    if len(keypoints) > 0:
        im_with_keypoints = cv2.resize(im_with_keypoints, (800, 500), interpolation=cv2.INTER_LINEAR)
        #cv2.imshow("Keypoints", im_with_keypoints)
        if len(keypoints) != lastKeypointsCount:
            lastKeypointsCount = len(keypoints)



depth_stream.stop()
openni2.unload()





 
 
  





