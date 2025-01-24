import cv2
import numpy as np
import os
import glob
from moviepy.editor import *

BASE_DIR_NAME = "img_stream"
DIMENSIONS = (1280, 720)
KERNEL_SIZE = (3, 3)
FILE_TYPE = ".png"

def main(reference_vid, filmed_vid, output_filename):
	extract_audio(reference_vid)
	video = cv2.VideoCapture(filmed_vid)
	ref = cv2.VideoCapture(reference_vid)
	ref_length = int(ref.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_rate = ref.get(cv2.CAP_PROP_FPS)
	img_count = 0
	broken = False

	# Create folder for storing the frames
	if not os.path.exists(BASE_DIR_NAME):
		os.mkdir(BASE_DIR_NAME)

	while True:
		ret, frame = video.read()
		ret2, refFrame = ref.read()

		# Check if videos are null
		if not ret or not ret2:
			break
	
		frame = cv2.resize(frame, DIMENSIONS)
		image = cv2.resize(refFrame, DIMENSIONS)
		
		# Set bounds for the greenscreen
		lower = np.array([0, 170, 0])
		upper = np.array([170, 255, 255])
		
		# Detect greenscreen bg using defined bounds
		mask = cv2.inRange(frame, lower, upper)

		# Apply morphology (dilate to lessen green pixels in border)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
		morph = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
		
		# Invert morphed mask to match object
		invert = 255 - morph

		# Get object
		object = cv2.bitwise_and(frame, frame, mask=invert)

		# Merge object with the reference video
		result = np.where(object == 0, image, object)

		cv2.imshow("frame", result)

		# Save frame as image
		path = f"{BASE_DIR_NAME}/{str(img_count).zfill(len(str(ref_length)))}{FILE_TYPE}"
		cv2.imwrite(path, result)

		print(f"Saving frame {path}")
		img_count += 1

		if cv2.waitKey(int(frame_rate)) == 27:
			broken = True
			break
	
	video.release()
	ref.release()
	cv2.destroyAllWindows()

	if not broken:
		# Compile saved frames into video
		compile(output_filename, frame_rate)
		add_audio(output_filename)
		os.remove("ref_audio.mp3")

# Function for creating the output video file from sequence of images
def compile(output_filename, frame_rate):
	# Use frame_rate to match the speed of the output and reference video (.avi)
	out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"DIVX"), frame_rate, DIMENSIONS)

	# Compile all frames into a video
	for filename in glob.glob(f"{os.getcwd()}\\{BASE_DIR_NAME}\\*{FILE_TYPE}"):
		frame = cv2.imread(filename)
		print(f"Compiling frame {filename}")
		out.write(frame)

	out.release()


def extract_audio(ref_vid):
	ref = VideoFileClip(ref_vid)
	ref.audio.write_audiofile(r"ref_audio.mp3")


def add_audio(result_vid):
	vid = VideoFileClip(result_vid)
	audioclip = AudioFileClip("ref_audio.mp3")
	output = vid.set_audio(audioclip)
	output.write_videofile("output.mp4")


main("reference_video.mp4", "input_video.mp4", "output.avi")

# References
# [https://www.geeksforgeeks.org/replace-green-screen-using-opencv-python/]
# [https://stackoverflow.com/questions/51719472/remove-green-background-screen-from-image-using-opencv-python]
# [https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python]
# [https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/]
# [https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd]
# [https://stackoverflow.com/questions/40445885/no-audio-when-adding-mp3-to-videofileclip-moviepy]