# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg
#import centroid_tracker
#from imutils.video import VideoStream
import cv2 as cv
import argparse
import sys
import time
import numpy as np
import os.path
import socket
import sys


class openCV:
    syringe = False
    def __init__(self, video_stream):
        self.video_stream = video_stream
        self.x_pixel = 0
        self.y_pixel = 0
        # Initialize the parameters
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold

        self.inpWidth = 416  # 608     #Width of network's input image
        self.inpHeight = 416  # 608     #Height of network's input image
        
        # Load names of classes
        self.classesFile = "yolo_files/classes.names";

        self.classes = None
        with open(self.classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

            # Give the configuration and weight files for the model and load the network using them.

        self.modelConfiguration = "yolo_files/syringe-yolov3-tiny.cfg";
        self.modelWeights = "yolo_files/syringe-yolov3-tiny_13000.weights";

        self.net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    # Get the names of the output layers
    def getOutputNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Draw the predicted bounding box
    def drawPred(self, classId, conf, left, top, right, bottom, frame):
        # Draw a bounding box.
        #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        
        label = '%.2f' % conf
        
        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                (0, 0, 255), cv.FILLED)
        #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        left = -1
        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            print("out.shape : ", out.shape)
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
                confidence = scores[classId]
                if detection[4] > self.confThreshold:
                    print(detection[4], " - ", scores[classId], " - th : ", self.confThreshold)
                    print(detection)
                  
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    self.set_x_and_y_pixels(center_x, center_y)
                    # print("center_x:", center_x)
                    # print("center_y:", center_y)
                    # print("here I am: ", boxes[0])
                

       # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)
           
        if left >= 0:
               self.syringe = True
        else:
            self.syringe = False
            self.set_x_and_y_pixels(-1, -1)
        print(self.syringe)
        print(self.x_pixel)
        print(self.y_pixel)

    def run(self):
        # Process inputs
        winName = 'Deep learning object detection in OpenCV'
        cv.namedWindow(winName, cv.WINDOW_NORMAL)

        outputFile = "yolo_out_py.avi"
    
        cap = cv.VideoCapture(self.video_stream)
        
        width = int(cap.get(3))
        heigth = int(cap.get(4))

        frame = cap.read()
        # Get the video writer initialized to save the output video
   
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        while cv.waitKey(1) < 0:
            # get frame from the video
           
            hasFrame, frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
 
                cv.waitKey(3000)
                break

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputNames(self.net))
            
            # Remove the bounding boxes with low confidence
            self.postprocess(frame, outs)

            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            vid_writer.write(frame.astype(np.uint8))

            cv.imshow(winName, frame)
            self.print_bool()

            robot = robotfloor(40, 1.0, 62.2 , 48.8, 3280, 1464)
            robot.set_pixels(self.x_pixel, self.y_pixel)
            robot.solve()

            #data = "fdasfdsafadsfsd"
            data = str(self.syringe) + ", " + str(robot.get_x_world()) + ", " + str(robot.get_y_world()) + ", " +  str(robot.get_z_world())
            # Create a socket (SOCK_STREAM means a TCP socket)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Connect to server and send data
                sock.connect(("10.16.0.230", 6969))
                sock.sendall(bytes(data + "\n", "utf-8"))
                #Receive data from the server and shut down
                sock.recv(1024)
                # client = robot_client(self.ip_address, self.port2)
                # client.data_to_send(self.syringe, robot.get_x_world(), robot.get_y_world(), robot.get_z_world())

            if cv.waitKey(1) == 27: 
                break  # esc to quit

        #time.sleep(0.5)

    def get_x_pixel(self):
        return self.x_pixel

    def get_y_pixel(self):
        return self.y_pixel

    def set_x_pixel(self, x_pixel):
        self.x_pixel = x_pixel - self.stream_width

    def set_y_pixel(self, y_pixel):
        self.y_pixel = y_pixel - self.stream_height

    def set_x_and_y_pixels(self, x_pixel, y_pixel):
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        #print("This the x pixel coordinate: ", self.x_pixel)

    def print_y_pixel(self):
        print("This the y pixel coordinate: ", self.y_pixel)

    def print_xy_coordinates(self):
        coordinates  = "[" + str(self.x_pixel) + "," + str(self.y_pixel) + "]"
        print('These are the x,y coordinates ', coordinates)

    def print_bool(self):
        print(f'{str(self.x_pixel)} {str(self.y_pixel)} {self.syringe}')
  
import math

class robotfloor:
    
    pixel_x = 0
    pixel_y = 0
    x_world = 0
    y_world = 0
    z_world = 0

    def __init__(self, beta, height, alpha_x, alpha_y, res_x, res_y):
        self.beta = math.radians(beta)
        self.height = height
        self.alpha_x = math.radians(alpha_x)
        self.alpha_y = math.radians(alpha_y)
        self.h_res_x = res_x / 2
        self.h_res_y = res_y / 2
        self.main_num = height * self.h_res_y
        self.tanHalf_of_Alpha = math.tan(self.alpha_y / 2)
        self.h_res_y__x__tanBeta = self.h_res_y * math.tan(self.beta)
        self.h_res_y__x__tanHalf_of_Alpha = self.h_res_y * self.tanHalf_of_Alpha
        self.main_dem = self.h_res_y__x__tanBeta - self.h_res_y__x__tanHalf_of_Alpha
        self.zm =  height / math.tan(self.beta)

    def set_pixels(self, pixel_x, pixel_y):
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def print_pixel_coordinates(self):
        print("xxxxxxx PIXEL XY COORDINATES xxxxxxx")
        print("xxxxxxx         %d,%d          xxxxxxx" % (self.pixel_x, self.pixel_y))


    def get_x_world(self):
        return self.x_world

    def get_y_world(self):
        return self.y_world

    def get_z_world(self):
        return self.z_world

    def string_xyz(self):
        return f'{str(self.x_world)} {str(self.y_world)} {str(self.z_world)}'



    def solve_z_world(self):

        #self.z_world = self.main_num / ((self.h_res_y__x__tanBeta - self.h_res_y__x__tanHalf_of_Alpha) + \
        #               (self.tanHalf_of_Alpha * self.pixel_y) )

        self.z_world = self.main_num / (self.main_dem + (self.tanHalf_of_Alpha * self.pixel_y))


    def solve_y_world(self):
        self.y_world = - self.height * (self.zm - self.z_world) / self.zm

    def solve_x_world(self):
        num = self.main_num* math.tan(self.alpha_x/2) * (self.h_res_x - self.pixel_x)
        dem1 = (self.h_res_y * math.tan(self.beta)) - (self.h_res_y * self.tanHalf_of_Alpha) + (self.pixel_y * self.tanHalf_of_Alpha)
        dem2 = self.h_res_x * dem1
        self.x_world = -1 * num / dem2

    def solve(self):

        self.solve_z_world()
        self.solve_y_world()
        self.solve_x_world()





video_feed = 'http://10.16.0.230:8090/test.mjpg'
test = openCV(video_feed)
test.run()
