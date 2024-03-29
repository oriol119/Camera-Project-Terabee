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
cv2.namedWindow("Filtered", cv2.WINDOW_NORMAL)

cv2.waitKey(90000)
depth_stream.stop()
openni2.unload()