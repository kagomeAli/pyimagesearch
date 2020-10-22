# USAGE
# python3 detect_faces.py --image images/rooster.jpg --prototxt model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel
# import 依赖包
import os
import numpy as np
import argparse
import cv2

# 构建参数（args）& 解析参数（args）
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 连接本地文件夹
dirname, filename = os.path.split(os.path.abspath(__file__))
prototxt = os.path.join(dirname, args["prototxt"])
model = os.path.join(dirname, args["model"])
image = os.path.join(dirname, args["image"])

# load our serialized model from disk
# 从disk中加载序列化磨model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# load the input image and construct an input blob for the image
# 加载输入的images & 将images转为二进制文件
# by resizing to a fixed 300x300 pixels and then normalizing it
# 调整图片大小 而后序列化图像
image = cv2.imread(image)
(h, w) = image.shape[:2]
print(image.shape[:2])
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, 
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
# 通过输出blob得到检查结果，并做预测
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
# 以上步骤已得到预测结果，已完成


# 显示图片，移动鼠标的过程显示鼠标在图像的位置
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)