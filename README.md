# FlynnyViz
In short, this script takes an input images and hue shifts it based off the current sound playing and a constant shift, making sort of an audio vizualizer

![RGB Flynn](examples/Demo.gif)

*Character from [Echo](https://echoproject.itch.io/echo) by [EchoProject](https://www.patreon.com/EchoGame), check them out if you haven't already!*

## Command Line Options
- -I, --image        Select image
- -T, --title        Allows you to change the name of the window (limited to ASCII)
- -s, --scale        Changes images scale in percentages
- -F, --fps_cap      Changes the default framerate cap
- --mask             Applies a grayscale mask to mask out areas (black - masked, white - unmasked)

Smoothing
- -D, --dampening     Scales the captured audio values in an inverse manner (higher dampening = lower values)
- --smoothing        Takes one capture every Nth frame and every N*2th frame to mix together
- -x1, --weightx1     Weight of Nth frame
- -x2, --weightx2     Weight of N*2 frame

-S, --stats           Displays FPS, real frame time, hue-shift and volume level

![Stats](examples/stats_example.png)
- --shift             Constant hue shift, helps with some problematic images
- --only_audio        Only shift hue based off of audio

### TODO
- Provide a packaged version of the script, for both ~~Windows~~ and Linux
- Implement input device selection
- Fix some weird display issues with openCV (maybe even switch to a different library)
- Add UI for image selection, live changes, etc.
- clean up the code ~~(yeah, no)~~
