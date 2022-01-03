#! python3
import threading
import os
import sys
from time import strftime
from queue import Queue, Empty

try:
    import UI_component
    import Base_component
    import argparse
except ImportError as error:
    sys.stdout.write(f'Error: "{error.name}" not installed or the files are missing\n')
    exit()


def FlynnyQuit():
    os._exit(1)


def argument_parse(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Hue shift images based off of sound playing on your default input device",
        epilog="Inspired by that RGB Flynn gif on the Echocord\nMade by D2#0621",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args_main = parser.add_argument_group()
    args_audio = parser.add_argument_group()
    args_misc = parser.add_argument_group()

    # arg init
    args_main.add_argument(
        "-I",
        "--image",
        type=str,
        metavar="",
        default=os.path.join(os.getcwd(), "img", "Flynn.png"),
        help="String with path to image",
    )
    args_main.add_argument(
        "-T",
        "--title",
        type=str,
        metavar="",
        default="FlynnyViz",
        help="Name of the window (only accepts ASCII characters)",
    )
    args_main.add_argument(
        "-F",
        "--fps-cap",
        type=int,
        metavar="",
        default=45,
        help="Change default framerate cap",
    )
    args_main.add_argument(
        "-s", "--scale", type=int, metavar="", default=100, help="Change image scale"
    )
    args_main.add_argument(
        "--mask", type=str, metavar="", default=None, help="Apply grayscale mask image"
    )

    args_audio.add_argument(
        "-D",
        "--dampening",
        type=float,
        metavar="",
        default=1.5,
        help="higher = less sensitivity to audio",
    )
    args_audio.add_argument(
        "--smoothing",
        type=int,
        metavar="",
        default="12",
        help="Takes one capture every Nth frame and every N*2th frame to mix together",
    )
    args_audio.add_argument(
        "-x1",
        "--weightx1",
        type=float,
        metavar="",
        default=0.7,
        help="Weight of Nth frame",
    )
    args_audio.add_argument(
        "-x2",
        "--weightx2",
        type=float,
        metavar="",
        default=0.4,
        help="Weight of N*2 frame",
    )

    args_misc.add_argument(
        "-S",
        "--stats",
        action="store_true",
        help="Displays FPS, real frame time, hue-shift and volume level",
    )
    args_misc.add_argument(
        "--only-audio", action="store_false", help="Only shift based off of audio"
    )
    args_misc.add_argument(
        "--shift",
        type=int,
        metavar="",
        default=30,
        help="Constant hue shift, helps with some problematic images",
    )
    args_misc.add_argument("--secret", action="store_true", help=argparse.SUPPRESS)

    return parser.parse_args(args)


if __name__ == "__main__":
    arguments = argument_parse()
    if arguments.secret:
        from stegano.lsb import reveal as dontlook

        sys.stdout.write(dontlook("img/Flynn.png")), exit(0)

    need_restart = 0
    queue = Queue(maxsize=1)
    # queue = Queue()

    try:
        print("Creating main thread...")
        main = threading.Thread(
            target=Base_component.main, args=[arguments, need_restart, queue]
        )
        print("Starting main thread")
        main.start()
    except Exception as err:
        print(err)
        with open('FlynnyViz.log', 'a') as log:
            log.write(f"{strftime('%d/%m/%Y %H:%M%S')} {repr(err)}\n")
        FlynnyQuit()

    try:
        print("Creating UI thread...")
        UI = threading.Thread(
            target=UI_component.UI, args=[FlynnyQuit, arguments, need_restart, queue]
        )
        print("Starting UI thread")
        UI.start()
    except Exception as err:
        with open("FlynnyViz.log", "a") as log:
            log.write(f"{strftime('%d/%m/%Y %H:%M%S')} {repr(err)}\n")
        FlynnyQuit()

    print("DONE!")
