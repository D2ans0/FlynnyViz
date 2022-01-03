import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttkB


class UI:
    filename = None
    srcimg = None

    def __init__(self, command="", arguments="", queue=None):
        self.shift = queue.shift
        self.img_name = queue.img_name
        self.arguments = arguments
        self.UI(command=command)

    # def testfunc(self, shift):
    #     self.queue.put(shift)

    def filesystem(self):
        self.filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select a File",
            filetypes=(
                ("CV2 supported images", "*.jpg* *.jpeg* *.png* *.bmp* *.dib* *.webp*"),
                ("all files", "*.*")
            )
        )

        if self.filename != "":
            self.img_name.put(self.filename)

    def UI(self, command=exit):
        window = tk.Tk()
        style = ttkB.Style(theme="FlynnyViz", themes_file="themes.json")

        # add widgets here
        window.title("FlynnyViz controls")
        window.geometry("350x200")
        base_shift = tk.ttk.Scale(
            window,
            from_=0,
            to=180,
            length=300,
            orient=tk.HORIZONTAL,
            command=self.shift.put
        )

        base_shift.pack(pady=10)

        browseIMG = tk.ttk.Button(window, text="Select image", command=self.filesystem)
        browseIMG.pack()

        if command != 0:
            window.protocol("WM_DELETE_WINDOW", command)

        window.mainloop()


if __name__ == "__main__":
    UI.UI(UI)
