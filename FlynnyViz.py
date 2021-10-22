#! python3

import os
import sys
import argparse
from time import time, sleep, strftime
try:
	import numpy as np
	import cv2
	import soundcard as sc
except ImportError as error:
	sys.stdout.write(f'Error: "{error.name}" not installed\n')
	exit()


# args
def argument_parse(args=sys.argv[1:]):
	parser = argparse.ArgumentParser(
		description="Hue shift images based off of sound playing on your default input device",
		epilog="Inspired by that RGB Flynn gif on the Echocord\nMade by D2#0621",
		formatter_class=argparse.RawDescriptionHelpFormatter)
	args_main = parser.add_argument_group()
	args_audio = parser.add_argument_group()
	args_misc = parser.add_argument_group()

	# arg init
	args_main.add_argument("-I", "--image", type=str, metavar='',
		default=os.path.join(os.getcwd(), "img", "Flynn.png"), help="String with path to image")
	args_main.add_argument("-T", "--title", type=str, metavar='',
		default="FlynnyViz", help="Name of the window (only accepts ASCII characters)")
	args_main.add_argument("-F", "--fps-cap", type=int, metavar='',
		default=45, help="Change default framerate cap")
	args_main.add_argument("-s", "--scale", type=int, metavar='',
		default=100, help="Change image scale")
	args_main.add_argument("--mask", type=str, metavar='',
		default=None, help="Apply grayscale mask image")

	args_audio.add_argument("-D", "--dampening", type=float, metavar='',
		default=1.5, help="higher = less sensitivity to audio")
	args_audio.add_argument("--smoothing", type=int, metavar='',
		default='12', help="Takes one capture every Nth frame and every N*2th frame to mix together")
	args_audio.add_argument("-x1", "--weightx1", type=float, metavar='',
		default=0.7, help="Weight of Nth frame")
	args_audio.add_argument("-x2", "--weightx2", type=float, metavar='',
		default=0.4, help="Weight of N*2 frame")

	args_misc.add_argument("-S", "--stats", action="store_true",
		help="Displays FPS, real frame time, hue-shift and volume level")
	args_misc.add_argument("--only-audio", action="store_false",
		help="Only shift based off of audio")
	args_misc.add_argument("--shift", type=int, metavar='', default=30,
		help="Constant hue shift, helps with some problematic images")
	args_misc.add_argument("--secret", action="store_true", help=argparse.SUPPRESS)

	return parser.parse_args(args)


# utils
def timer(time_count_start, mode=""):
	"""time_start - start time \n mode - time/rate/none(both)"""
	frame_time = time() - time_count_start
	if mode == "time":
		return frame_time

	else:
		if frame_time:
			frame_rate = int(1. / frame_time)
		else:
			frame_rate = "inf"
		if mode == "rate":
			return frame_rate
		else:
			return frame_time, frame_rate

	pass


class Flynny_image:
	info_x = 5
	info_y = 15
	info_scale = 0.4
	info_color = (255, 255, 255)

	@staticmethod
	def open_image(img_source_input):	# image to numpy array, allows for unicode characters in path
		with open(img_source_input, "rb") as image_stream:
			image_array = bytearray(image_stream.read())
		image_NParray = np.asarray(image_array, dtype=np.uint8)
		image_output = cv2.imdecode(image_NParray, -1)
		return image_output
	@staticmethod
	def scaling(image, percentage):
		percentage = percentage / 100
		target_size = (int(image.shape[1]*percentage), int(image.shape[0]*percentage))
		return cv2.resize(image, target_size)

	@staticmethod
	def apply_alpha(image, a = np.array([]), retrieve_alpha=False):
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
			if not retrieve_alpha:
				b,g,r,a = image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3]
				b,g,r = np.bitwise_and(a, b), np.bitwise_and(a, g), np.bitwise_and(a, r)
				return np.dstack([b,g,r,a])
			else:
				a = image[:,:,3]
				return a

	@staticmethod
	def info_overlay(self, image, info):
		info = "\n".join((f"fps: {info['fps']} Frame time (ms): {(info['time']*100):.1f}",
			f"Shift: {info['dyn_shift']} (Base: {info['base_shift']})",
			f"Volume: Real: {info['vol_real']:.0f} Smoothed: {info['vol_smoothed']:.0f}",
			f"{info['mic']}"))
		for i, line in enumerate(info.splitlines()):
			cv2.putText(image, line,
				(self.info_x, self.info_y + int(38 * self.info_scale * i)),
				cv2.FONT_HERSHEY_SIMPLEX,
				self.info_scale,
				self.info_color)


class Flynny_audio:
	@staticmethod
	def get_mic():
		def_speaker = sc.default_speaker()
		def_speaker = str(def_speaker)
		def_speaker = def_speaker[def_speaker.find("("):def_speaker.find(")") + 1]
		mic = sc.get_microphone(def_speaker, include_loopback=True)
		return mic
	@staticmethod
	def audio_sens(audio_stream, dampening):
		data = audio_stream.record(numframes=16)
		data_merged = (np.max(data) - np.min(data)) * 1000 / dampening
		return data_merged


if __name__ == "__main__":
	def main(audio_stream):
		# var init
		fps_cap = 1 / args.fps_cap
		frame_time, info, iterator, volume_2, volume_3 = time(), 0, 0, 0, 0

		# open img and scale/apply alpha when needed
		img_src = Flynny_image.open_image(args.image)

		if args.scale != 100: img_src = Flynny_image.scaling(img_src, args.scale)
		if img_src.shape[2] == 4: alpha = Flynny_image.apply_alpha(img_src, retrieve_alpha=True)
		else: alpha = np.array([])

		# BGR -> HSV and split channels
		imgHSV = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
		h,s,v = imgHSV[:,:,0], imgHSV[:,:,1], imgHSV[:,:,2]

		# area masking
		if args.mask:
			mask = Flynny_image.open_image(args.mask)
			if args.scale != 100: mask = Flynny_image.scaling(mask, args.scale)
			if mask.shape[2] >= 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			masked = np.bitwise_or(mask, h)

		while cv2.getWindowProperty(args.title, cv2.WND_PROP_VISIBLE):  # run till main WND is closed
			timerStart = time()
			iterator = (iterator + 1) % 180
			volume_real = Flynny_audio.audio_sens(audio_stream, args.dampening)

			# smoothing
			if not (iterator % args.smoothing): volume_2 = volume_real * args.weightx1
			if not (iterator % (args.smoothing*2)): volume_3 = volume_real * args.weightx2
			volume_smoothed = (volume_real + volume_2 + volume_3) / (1 + args.weightx1 + args.weightx2)  # average out loudness

			# calculate amount of hue shift and modify the channel
			shift = args.shift + int(volume_smoothed + iterator * args.only_audio) % 180
			h_shifted = shift - h

			# apply hue shift and convert back to BGR
			if not args.mask: imgHSV = np.dstack([h_shifted, s, v])
			else:
				h_shifted = np.bitwise_or(np.invert(mask), h_shifted)
				imgHSV = np.dstack([np.bitwise_and(masked, h_shifted), s, v])
			img_HSV2BGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

			# check if there's an alpha channel and either apply it or display as is
			if alpha.size == 0: img_out = img_HSV2BGR
			else: img_out = Flynny_image.apply_alpha(img_HSV2BGR, alpha)

			# stats overlay
			if args.stats and info: Flynny_image.info_overlay(img_out, info)

			cv2.imshow(args.title, img_out)
			cv2.waitKey(1)

			frame_time = timer(timerStart, mode="time")
			sleep_time = fps_cap - frame_time
			if sleep_time >= 0: sleep(sleep_time)
			fps = timer(timerStart, mode='rate')
			info = {'fps': fps,
				'time': frame_time,
				'dyn_shift': shift,
				'base_shift': iterator,
				'vol_real': volume_real,
				'vol_smoothed': volume_smoothed,
				'mic': micname}

	args = argument_parse()
	if args.secret:
		from stegano.lsb import reveal as dontlook
		sys.stdout.write(dontlook("img/Flynn.png")), exit(0)

	cv2.namedWindow(args.title, flags=(cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE))
	mic = Flynny_audio.get_mic()
	micname = str(mic)
	try:
		with mic.recorder(samplerate=1000, blocksize=16) as stream:
			main(stream)

	except Exception as err:
		with open('FlynnyViz.log', 'a') as log:
			log.write(f"{strftime('%d/%m/%Y %H:%M%S')} {repr(err)}\n")
		with mic.recorder(samplerate=1000, blocksize=16) as stream:
			main(stream)
