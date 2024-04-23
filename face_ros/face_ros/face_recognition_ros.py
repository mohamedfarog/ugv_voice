#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import face_recognition
import numpy as np
 

class Face(object):
	def __init__(self):
		
		# Load a second sample picture and learn how to recognize it.

		self.wuguo_image = face_recognition.load_image_file("./src/face_ros/face_ros/pictures/M.FAROG.png")
		self.wuguo_face_encoding = face_recognition.face_encodings(self.wuguo_image)[0]
		# Create arrays of known face encodings and their names
		self.known_face_encodings = [self.wuguo_face_encoding]
		self.known_face_names = ["M.FAROG"]
		# Initialize some variables
		self.face_locations = []
		self.face_encodings = []
		self.face_names = []
		self.process_this_frame = True

		self.data = 0.0

		
	def	faceConfidence(self, face_match_threshold=0.6):
		range = (1.0 - self.data)
		linear_val = (1.0 - self.data) / (range * 2.0)

		if self.data > face_match_threshold:
			return round(linear_val * 100, 2)
		else:
			value = (linear_val + ((1.0 - linear_val) * (((linear_val - 0.5) * 2)
				** 0.2))) * 100
		return round(value, 2)

	

	def detector(self, img):
		while True:
		# Resize frame of video to 1/4 size for faster face recognition processing
			small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
		
			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			# rgb_small_frame = small_frame[:, :, ::-1]
			rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

		# Only process every other frame of video to save time
			if self.process_this_frame:
			# Find all the faces and face encodings in the current frame of video
				self.face_locations = face_recognition.face_locations(rgb_small_frame)
				self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

				#face_names = []
				for face_encoding in self.face_encodings:
					# See if the face is a match for the known face(s)
					matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
					name = "Unknown"

					# Or instead, use the known face with the smallest distance to the new fac
					self.face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
					best_match_index = np.argmin(self.face_distances)
					if matches[best_match_index]:
						range = 0.4
						linear_val = (1.0 - self.face_distances[best_match_index]) / (range * 2.0)
						result = 0.0

						if self.face_distances[best_match_index] > 0.6:
							result = round(linear_val * 100, 2)
						else:
							value = (linear_val + ((1.0 - linear_val) * (((linear_val - 0.5) * 2)
								** 0.2))) * 100
							result = round(value, 2)
						name = 'Unknown' if result < 94.0 else self.known_face_names[best_match_index]
					self.face_names.append(name)


			#self.process_this_frame = not self.process_this_frame
			return self.face_locations, self.face_names
		
class face_recognition_ros(Node):

	def __init__(self):
		super().__init__('face_recognition_ros')
		self.bridge = CvBridge()
		self.image_pub = self.create_publisher(Image, "/face_detection/face_image", 10)
		self.image_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.callback,10)

	def callback(self,data):
		try:
			cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print (e)
	
		self.face = Face()
		face_locations, face_names = self.face.detector(cv_img)
		print("face_names:",face_names)
		# Display the results
		for (top, right, bottom, left), name in zip(face_locations, face_names):
			# Scale back up face locations since the frame we detected in was scaled to 1/4 size
			top *= 4
			right *= 4 
			bottom *= 4
			left *= 4

			# Draw a box around the face
			cv2.rectangle(cv_img, (left, top), (right, bottom), (255, 0, 0), 2)

			# Draw a label with a name below the face
			cv2.rectangle(cv_img, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(cv_img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	   # Display the resulting image
		cv2.imshow('Video', cv_img)
		cv2.waitKey(3)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
		except CvBridgeError as e:
			print (e)

def main(args=None):
	rclpy.init(args=args)

	face_ros = face_recognition_ros()

	rclpy.spin(face_ros)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	image_converter.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()