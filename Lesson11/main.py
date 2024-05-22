#importing libraries
import cv2
import matplotlib.pyplot as plt
from matplotlib import ft2font
print("Libraries imported successfully!")

#importing and using necessary files
config_file='Lesson11/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='Lesson11/frozen_inference_graph.pb'

#Tenserflow object detection model
model = cv2.dnn_DetectionModel(frozen_model,config_file)

#Reading Coco dataset
classLabels=[]
filename='Lesson11/yolo3.txt'
with open(filename,'rt') as fpt:
  classLabels = fpt.read().rstrip('\n').split('\n')

print("Number of Classes")
print(len(classLabels))
print("Class labels")
print(classLabels)


#Model training
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#reading image
img = cv2.imread('Lesson11/people.jpeg')
cv2.imshow("real",img)
cv2.waitKey(0)

#converting image from BGR to RGB
cv2.imshow("rgb",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
#object detection
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

#fetching accuracy
print(confidence)

#fetching object index
print(ClassIndex)

#plotting boxes
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (0, 255, 0), 3)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 0, 255), thickness=3)
cv2.imshow("final",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)