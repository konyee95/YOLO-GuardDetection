# USAGE 
# python3  mongo.py --yolo yolo-guard

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import datetime
import cv2
import os
import pymongo
from pymongo import MongoClient

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the GUARD class labels YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"],"guards.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny_37856.weights"])
configPath =  os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])

# load our YOLO object detector trained on GUARD dataset
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

myClient = MongoClient('localhost',27017)
db = myClient.TG_database
collection = db.date_collection

class Guard:
    def __init__(self):
        self.dateTime = None
        self.firstSeen = None
        self.lastSeen = None
        self.disappearedCount = 0
        self.gotGuard = False

guard = Guard()
seenHistory = list()

# first set no guard is detected
detectGuard = False

# rescale video frame to get a better video streaming view
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# initialize the video stream, pointer to output video file
vs = cv2.VideoCapture()
vs.open("rtsp://it:admin123@10.2.13.11:554/Streaming/Channels/101/")
(W,H) = (None, None)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    frame65 = rescale_frame(frame, percent = 65)

    # if the frame was not grabbed, then we have reached of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame65.shape[:2]
    
    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame65, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each detections
        for detection in output:
            # extract the class ID and confidence (ie., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
				# confidences, and class ID
                boxes.append([x,y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
    # apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, args["confidence"], args["threshold"])
    
    # ensure at lest one detection exists
    if len(idxs) > 0:
        # check if previously is not detected
        if guard.gotGuard == False:
            guard.firstSeen = time.time()
            guard.gotGuard = True

        # update latest time seeing guard
        guard.lastSeen = time.time()
        print("Last Seen Time: ",guard.lastSeen)

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extraxt the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # get how long the guard appeared on camera
            guardSeenDuration = guard.lastSeen - guard.firstSeen
            print("Current time: {}".format(guard.lastSeen))
            print("Guard Seen Duration: {}".format(guardSeenDuration))

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame65, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f} {}".format(LABELS[classIDs[i]], confidences[i], (guardSeenDuration))
            cv2.putText(frame65, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            gdata = {"time":datetime.datetime.now() , "guard_exist":True, "time_duration":guardSeenDuration}
            db_data = collection.insert_one(gdata)
            print(db_data.inserted_id)
    
    else:
        # no guard is detected, increase count
        guard.disappearedCount += 1

        # guard disappeared for too long, record down history
        if guard.disappearedCount > 200:
            first, last = guard.firstSeen, time.time()
            seenHistory.append((first,last))

            # reset guard
            guard.gotGuard = False
            guard.firstSeen = None
            guard.disappearedCount = 0

    cv2.imshow("frame", frame65)
    key = cv2.waitKey(1)
    