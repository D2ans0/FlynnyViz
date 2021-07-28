#! python3

from os import system, path, truncate
import argparse
import numpy as np
from time import time, sleep
import cv2
from pyaudio import PyAudio, paInt16

### flags ###
# add args
args_setup = argparse.ArgumentParser(description="Hue shift image based off of sound playing on default input device", epilog="Inspired by that RGB Flynn gif on the Echocord\nMade by D2#0621")
args_audio = args_setup.add_argument_group()
args_misc = args_setup.add_argument_group()

args_setup.add_argument("-I", "--image", type=str, metavar='', required=False, default=r"Flynn.png", help="String with path to image")
args_setup.add_argument("-T", "--title", type=str, metavar='', default="FlynnyViz", help="Name of the window (only accepts ASCII characters)")
args_setup.add_argument("-F","--fps_cap", type=int, metavar='', default=45, help="Change default framerate cap")

args_audio.add_argument("-D", "--dampening", type=int, metavar='', default=16, help="higher = less sensitivity to audio")
args_audio.add_argument("-t", "--timing", type=int, metavar='', default='6', help="Takes one capture every Nth frame and every N*2th frame to mix together")
args_audio.add_argument("-x1", "--weightx1", type=float, metavar='', default=0.7, help="Weight of Nth frame")
args_audio.add_argument("-x2", "--weightx2", type=float, metavar='', default=0.4, help="Weight of N*2 frame")

args_misc.add_argument("-S", "--stats", action="store_true", help="Displays FPS, real frame time, hue-shift and volume level")
args_misc.add_argument("--shift", type=int, metavar='', default=15, help="Constant hue shift, helps with some problematic images")
if path.splitext(__file__)[1] != ".pyw":
	args_misc.add_argument("--noclear", action="store_true", help="Do not clear console on window close (may help with debugging)")
args_misc.add_argument("--secret", action="store_true", help=argparse.SUPPRESS)

# apply args
args = args_setup.parse_args()
window_title = args.title
image_path = args.image
framerate_cap = args.fps_cap

const_shift = args.shift
dampening = args.dampening

smoothing_timing = args.timing
smoothing_weight_x1 = args.weightx1
smoothing_weight_x2 = args.weightx2

stats = args.stats

secret = args.secret
#nothing to see here
if secret: print("LS4tLSAtLS0gLi4tIC4tLS0tLiAuLS4gLiAvIC4tIC8gLS4uLiAtLS0gLSAtIC0tLSAtLQ=="), exit(0)

### main variables ###
volume_2, volume_3, iterator, fps, frame_time = 0, 0, 0, 0, 0
framerate_cap = 1/framerate_cap #calculate nominal frame time

font_pos = (5, 15)
font_scale = 1
font_Face = cv2.FONT_HERSHEY_SIMPLEX
font_weight = 1
font_color = (255,255,255)


max_value = 2**dampening
stream = PyAudio().open(
	format=paInt16,
	channels=1,
	rate=44100,
	input=True,
	frames_per_buffer=256)


### main functions ###
def open_image(image_name):	#image to numpy array, allows the use unicode characters in path
	image_stream = open(image_path, "rb")
	image_array = bytearray(image_stream.read())
	image_NParray = np.asarray(image_array, dtype=np.uint8)
	image_output = cv2.imdecode(image_NParray, -1)
	return image_output

def fps_calc(time_count_start, mode=""):	#1st var = fps, 2nd var = frame time
	frame_time = time() - time_count_start
	if frame_time:
		frame_rate = int(1. / frame_time)
	else:
		frame_rate = "inf"

	if mode == "rate":
		return frame_rate
	elif mode == "time":
		return frame_time
	else:
		return frame_rate, frame_time

def audio_sens(audio_stream):	
	data = np.frombuffer(audio_stream.read(1024),dtype=np.int16)
	data_merged = (np.max(data)-np.min(data))/max_value
	return data_merged

def apply_alpha(image_name, a = np.array([]), alphaOnly=False):
	if a.any() != 0:
		b,g,r = image_name[:,:,0], image_name[:,:,1], image_name[:,:,2] # split image into separate channels
		b,g,r = np.bitwise_and(a, b), np.bitwise_and(a, g), np.bitwise_and(a, r) # combine alpha and color channels, later merge
		return np.dstack([b,g,r,a])
	else:
		if not alphaOnly:
			b,g,r,a = image_name[:,:,0], image_name[:,:,1], image_name[:,:,2], image_name[:,:,3]
			b,g,r = np.bitwise_and(a, b), np.bitwise_and(a, g), np.bitwise_and(a, r)
			return np.dstack([b,g,r,a])
		else:
			a = image_name[:,:,3]
			return a
if stats:
	def info_overlay(image_name, x, y, scale, face, color):
		cv2.putText(image_name,f"fps: {fps} Frame time (ms): {round(frame_time*100, 2)}", (x,y), face, scale, color)
		cv2.putText(image_name,f"Shift: {shift} (Base: {iterator})", (x,y*2), face, scale, color)
		cv2.putText(image_name,f"Volume: Real: {int(volume_real)} Smoothed: {int(volume_smoothed)}", (x,y*3), face, scale, color)
		# print general info
		print("{}  \nfps: {}  \t\t(Frame time: {})  \t|\nShift: {}  \t\tBase: {}  \t\t|\nVolume: Real: {} \tSmoothed: {} \t\t|".format(window_title, fps, round(frame_time*100, 2), shift, iterator,int(volume_real), int(volume_smoothed)))
		system("echo \x1B[5A")
else: 
	def info_overlay(*_): return

img_source = open_image(image_path) #read image

# check if image has alpha
if img_source.shape[2] == 4:
	alpha = apply_alpha(img_source, alphaOnly=True)
else:
	alpha = np.array([])

imgHSV = cv2.cvtColor(img_source, cv2.COLOR_BGR2HSV)
h,s,v = imgHSV[:,:,0], imgHSV[:,:,1], imgHSV[:,:,2]

cv2.namedWindow(window_title) #init main window

while cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE):	# mainloop, runs as long as the main window running
	startTime0 = time()
	iterator = (iterator + 1) % 180
	volume_real = audio_sens(stream)*1000

	# smoothing
	if not (iterator % smoothing_timing): volume_2 = volume_real*smoothing_weight_x1
	if not (iterator % smoothing_timing*2): volume_3 = volume_real*smoothing_weight_x2
	volume_smoothed = (volume_real + volume_2 + volume_3) / (1 + smoothing_weight_x1 + smoothing_weight_x2) #average out loudness

	# calculate amount of hue shift and modify the channel
	shift = const_shift + int(volume_smoothed - iterator) % 180
	h_shifted = shift - h

	# apply hue shift and convert back to BGR
	imgHSV = np.dstack([h_shifted,s,v])
	img_HSV2BGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

	# check if there's an alpha channel and either apply it or display as is
	if alpha.any() == False:
		img_out = img_HSV2BGR
	else:
		alpha
		img_out = apply_alpha(img_HSV2BGR, alpha)

	# add info overlay
	info_overlay(img_out, 5, 15, 0.4, cv2.FONT_HERSHEY_SIMPLEX, (255,255,255))

	cv2.imshow(window_title, img_out) #redraw main window
	cv2.waitKey(1) #wait 1ms (required by opencv)

	# fps capping
	frame_time = fps_calc(startTime0, mode="time")
	sleep_time = (framerate_cap - frame_time)
	if sleep_time >= 0:
		sleep(sleep_time)
	fps = fps_calc(startTime0, mode="rate")

if not args.noclear: system("cls")