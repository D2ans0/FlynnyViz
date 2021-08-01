#! python3

import os 
import sys
import argparse
from time import time, sleep
try:
	import numpy as np
	import cv2
except ImportError as error:
	sys.stdout.write(f'"{error.name}" is not installed, you can install it with:\npip install {error.name}\n')
	exit()

### functions ###
#args
def argument_parse(args=sys.argv[1:]):
	parser = argparse.ArgumentParser(
		description="Hue shift images based off of sound playing on your default input device",
		epilog="Inspired by that RGB Flynn gif on the Echocord\nMade by D2#0621",
		formatter_class=argparse.RawDescriptionHelpFormatter)
	group1 = parser.add_argument_group()
	group2 = parser.add_argument_group()
	group3 = parser.add_argument_group()
	main_args(group1)
	audio_args(group2)
	misc_args(group3)
	return parser.parse_args(args)
def main_args(group):
	group.add_argument("-I", "--image", type=str, metavar='',
		default=rf"{os.getcwd()}/img/Flynn.png", help="String with path to image")
	group.add_argument("-T", "--title", type=str, metavar='',
		default="FlynnyViz", help="Name of the window (only accepts ASCII characters)")
	group.add_argument("-F","--fps-cap", type=int, metavar='',
		default=45, help="Change default framerate cap")
	group.add_argument("-s","--scale", type=int, metavar='',
		default=100, help="Change image scale")
	group.add_argument("--mask", type=str, metavar='',
		default=None, help="Apply grayscale mask image")
def audio_args(group):
	group.add_argument("-L", "--list-devices", action="store_true")
	group.add_argument("-D", "--dampening", type=int, metavar='',
		default=17, help="higher = less sensitivity to audio")
	group.add_argument("-t", "--timing", type=int, metavar='',
		default='6', help="Takes one capture every Nth frame and every N*2th frame to mix together")
	group.add_argument("-x1", "--weightx1", type=float, metavar='',
		default=0.7,help="Weight of Nth frame")
	group.add_argument("-x2", "--weightx2", type=float, metavar='',
		default=0.4, help="Weight of N*2 frame")
def misc_args(group):
	group.add_argument("-S", "--stats", action="store_true",
		help="Displays FPS, real frame time, hue-shift and volume level")
	group.add_argument("--shift", type=int, metavar='', default=15,
		help="Constant hue shift, helps with some problematic images")
	#if os.path.splitext(__file__)[1] != ".pyw":
	group.add_argument("--noclear", action="store_true",
		help="Do not clear console on window close (may help with debugging)")
	#
	group.add_argument("--only-audio", action="store_false",
		help="Only shift based off of audio")
	group.add_argument("--secret", action="store_true", help=argparse.SUPPRESS)

#utils
def open_image(image):	#image to numpy array, allows the use unicode characters in path
	with open(image, "rb") as image_stream:
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

def audio_stream():
	global stream, sd, audio_sens
	try:
		import sounddevice as sd
		stream = sd.InputStream(dtype=np.int16 , channels=1, samplerate=4000, blocksize=64)
		stream.start()
		def audio_sens(audio_stream):
			data = audio_stream.read(64)[0]
			data_merged = (np.max(data) - np.min(data))*1000/max_value
			return data_merged
	except ImportError:
		try:
			from pyaudio import PyAudio, paInt16
			stream = PyAudio().open(format=paInt16, channels=1, rate=4000, input=1, frames_per_buffer=64)
			def audio_sens(audio_stream):	
				data = np.frombuffer(audio_stream.read(64),dtype=np.int16)
				data_merged = (np.max(data) - np.min(data))*1000/max_value
				return data_merged

		except ImportError:
			sys.stdout.write(
				"""Neither pyaudio nor sounddevice were detected, try installing the latter with:
 pip install sounddevice""")
			exit()

def scaling(image, percentage):
	percentage = percentage / 100
	target_size = (int(image.shape[1]*percentage), int(image.shape[0]*percentage))
	return cv2.resize(image, target_size)

def apply_alpha(image, a = np.array([]), alphaOnly=False):
	if a.any() != 0:
		if image.shape[2] >= 4:
			b,g,r,_ = image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3] # split image into separate channels
			b,g,r = np.bitwise_and(a, b), np.bitwise_and(a, g), np.bitwise_and(a, r) # combine alpha and color channels, later merge
			return np.dstack([b,g,r,a])
		else: 
			b,g,r = image[:,:,0], image[:,:,1], image[:,:,2] # split image into separate channels
			b,g,r = np.bitwise_and(a, b), np.bitwise_and(a, g), np.bitwise_and(a, r) # combine alpha and color channels, later merge
			return np.dstack([b,g,r,a])
	else:
		if not alphaOnly:
			b,g,r,a = image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3]
			b,g,r = np.bitwise_and(a, b), np.bitwise_and(a, g), np.bitwise_and(a, r)
			return np.dstack([b,g,r,a])
		else:
			a = image[:,:,3]
			return a

def info_overlay(image, x, y, scale, face, color, info):
	"""some info"""
	cv2.putText(image, f"fps: {info[0]} Frame time (ms): {(info[1]*100):.1f}", (x,y), face, scale, color)
	cv2.putText(image, f"Shift: {info[2]} (Base: {info[3]})", (x,y*2), face, scale, color)
	cv2.putText(image, f"Volume: Real: {info[4]:.0f} Smoothed: {info[5]:.0f}", (x,y*3), face, scale, color)
	# print general info
	sys.stdout.write(f"{args.title}  \nfps: {info[0]}  \t\t(Frame time: {info[1]:.2f})  \t|\nShift: {info[2]}  \t\tBase: {info[3]}  \t\t|\nVolume: Real: {info[4]:.0f} \tSmoothed: {info[5]:.0f} \t\t|\n\x1B[4A")

def mainloop():
	global iterator, volume_2, volume_3, info
	cv2.namedWindow(args.title) #init main window
	while cv2.getWindowProperty(args.title, cv2.WND_PROP_VISIBLE):	# mainloop, runs as long as the main window running
		startTime0 = time()
		iterator = (iterator + 1) % 180
		volume_real = audio_sens(stream)

		# smoothing
		if not (iterator % args.timing): volume_2 = volume_real*args.weightx1
		if not (iterator % args.timing*2): volume_3 = volume_real*args.weightx2
		volume_smoothed = (volume_real + volume_2 + volume_3) / (1 + args.weightx1 + args.weightx2) #average out loudness

		# calculate amount of hue shift and modify the channel
		shift = args.shift + int(volume_smoothed - iterator*args.only_audio) % 180
		h_shifted = shift - h

		# apply hue shift and convert back to BGR
		if not args.mask: imgHSV = np.dstack([h_shifted,s,v])
		else:
			h_shifted = np.bitwise_or(np.invert(mask), h_shifted),
			imgHSV = np.dstack([np.bitwise_and(masked, h_shifted),s,v])
		img_HSV2BGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

		# check if there's an alpha channel and either apply it or display as is
		if alpha.any() == False:
			img_out = img_HSV2BGR
		else:
			#alpha
			img_out = apply_alpha(img_HSV2BGR, alpha)

		# add info overlay
		if args.stats: info_overlay(img_out, 5, 15, 0.4, cv2.FONT_HERSHEY_SIMPLEX, (255,255,255), info)

		cv2.imshow(args.title, img_out) #redraw main window
		cv2.waitKey(1) #wait 1ms (required by opencv)

		# fps capping
		frame_time = fps_calc(startTime0, mode="time")
		sleep_time = args.fps_cap - frame_time
		if sleep_time >= 0:
			sleep(sleep_time)
		fps = fps_calc(startTime0, mode="rate")
		info = (fps, frame_time, shift, iterator, volume_real, volume_smoothed)

	if not args.noclear: os.system("cls")

### root ###
if __name__ == "__main__":
	args = argument_parse()
	audio_stream()
	if args.list_devices: print('Available devices:\n', sd.query_devices()), exit()
	#nothing to see here
	if args.secret:
		from stegano.lsb import reveal as dontlook
		sys.stdout.write(dontlook("img/Flynn.png")), exit(0)
	
	#variables setup
	volume_2, volume_3, iterator = 0, 0, 0
	info = (0, 0, 0, 0, 0, 0)
	args.fps_cap = 1/args.fps_cap #calculate nominal frame time
	font_pos, font_scale, font_weight = (5, 15), 1, 1
	font_color, font_Face = (255,255,255), cv2.FONT_HERSHEY_SIMPLEX
	max_value = 2**args.dampening

	img_source = open_image(args.image) 						#read image
	if args.scale != 100: img_source = scaling(img_source, args.scale)	#scale on demand
	# check if image has alpha
	if img_source.shape[2] == 4:
		alpha = apply_alpha(img_source, alphaOnly=True)
	else:
		alpha = np.array([])

	#convert to HSV and split channels
	imgHSV = cv2.cvtColor(img_source, cv2.COLOR_BGR2HSV)
	h,s,v = imgHSV[:,:,0], imgHSV[:,:,1], imgHSV[:,:,2]

	#check for mask and apply if needed
	if args.mask:
		mask = open_image(args.mask)
		if args.scale != 100: mask = scaling(mask, args.scale)
		if mask.shape[2] >= 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		masked = np.bitwise_or(mask, h)
	mainloop()