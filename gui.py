import os
from queue import Queue
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2
import toml
from PIL import ImageTk
from PIL.Image import fromarray
from tkvideo import tkvideo
from videoProcesser import VideoProcessor

with open('configuration.toml', 'r') as f:
    config = toml.load(f)


class FoosGUI:
    def __init__(self, conf):
        self.filepath = None
        self.video = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.frameCount = 0
        self.rect_points = []
        self.i = 1
        self.start_playing = False
        self.queue = None
        self.progress_queue = Queue()
        self.finishedProc = False
        self.config = conf
        # Initialize main window
        self.win = Tk()
        self.win.geometry("1200x800")
        self.win.title("foosball-app")
        self.win.config(background="lightgrey")

        self.main_frame = Frame(self.win, bg="lightgrey")
        self.main_frame.pack(fill=BOTH, expand=YES)

        self.create_file_browser()
        self.win.mainloop()

    def create_file_browser(self):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # File browser UI
        label_file_explorer = Label(
            self.main_frame,
            text="File Explorer",
            width=80,
            height=2,
            bg="lightgrey"
        )
        button_explore = Button(
            self.main_frame,
            text="Browse Files",
            command=lambda: self.browseFiles(label_file_explorer),
            width=50,
            height=4,
            bg="grey"
        )
        label_file_explorer.pack(pady=10)
        button_explore.pack(pady=20)

    def browseFiles(self, label_file_explorer):
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select a File",
            filetypes=(
                ("Video files", "*.mp4;*.avi"),
                ("All files", "*.*")
            )
        )
        if filename:
            label_file_explorer.configure(text="File Opened: " + filename)
            self.filepath = filename
            self.plainSelector()

    def plainSelector(self):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.video = cv2.VideoCapture(self.filepath)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frameCount = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

        if self.filepath is None:
            print("No file selected")
            return

        ret, frame = self.video.read()
        if not ret:
            print("Failed to read video frame")
            return

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        canvas = Canvas(self.main_frame, width=imgtk.width(), height=imgtk.height())
        canvas.pack(expand=YES, fill=BOTH)
        canvas.image = imgtk
        canvas.create_image(0, 0, anchor=NW, image=imgtk)

        def restart():
            canvas.delete("all")
            canvas.image = imgtk
            canvas.create_image(0, 0, anchor=NW, image=imgtk)
            self.i = 1
            self.rect_points = []

        def start():
            self.start_playing = True
            self.queue = Queue(self.frameCount)
            vp = VideoProcessor(self.video, self.rect_points, self.queue, self.win, self.progress_queue, config)

            # Show progress bar in main frame
            for widget in self.main_frame.winfo_children():
                widget.destroy()

            progress_label = Label(self.main_frame, text="Processing video...", bg="lightgrey")
            progress_label.pack(pady=10)

            progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate")
            progress_bar.pack(pady=10)

            length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar["maximum"] = length

            def update_progress():
                try:
                    i = self.progress_queue.get_nowait()
                    progress_bar["value"] = i + 1
                    if i >= length - 1:  # Processing complete
                        print("Processing complete")
                        vp.stop_processing()  # Stop the processing thread
                        processed_video_path = "./processed.mp4"
                        if os.path.exists(processed_video_path):  # Check if the video exists
                            self.play_processed_video(processed_video_path)
                        else:
                            print("Processed video file not found.")
                    else:
                        self.win.after(100, update_progress)
                except Exception:
                    self.win.after(100, update_progress)

            update_progress()

        def paint(event):
            if self.i > 4:
                return
            green = "#ccffcc"
            x1, y1 = (event.x - 5), (event.y - 5)
            x2, y2 = (event.x + 5), (event.y + 5)
            if self.i == 4:
                canvas.create_oval(x1, y1, x2, y2, width=10, fill='white', outline=green)
                canvas.create_line(self.rect_points[self.i - 2][0], self.rect_points[self.i - 2][1], event.x, event.y,
                                   fill="green", width=5)
                canvas.create_line(self.rect_points[0][0], self.rect_points[0][1], event.x, event.y, fill="green",
                                   width=5)
                self.rect_points.append((event.x, event.y))
                self.i += 1
                return
            if self.i > 1:
                canvas.create_line(self.rect_points[self.i - 2][0], self.rect_points[self.i - 2][1], event.x, event.y,
                                   fill="green", width=5)

            canvas.create_oval(x1, y1, x2, y2, width=10, fill='white', outline=green)
            self.i += 1
            self.rect_points.append((event.x, event.y))

        canvas.bind("<Button-1>", paint)
        button_validate = Button(self.main_frame, text="Finish", command=start)
        button_validate.pack(side=LEFT, padx=10, pady=10)

        button_restart = Button(self.main_frame, text="Restart", command=restart)
        button_restart.pack(side=RIGHT, padx=10, pady=10)

    def play_processed_video(self, processed_video_path):
        if self.config['gui']['video_after_processing']:
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            self.main_frame.destroy()

            cap = cv2.VideoCapture(processed_video_path)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Processed Video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            # video_label = Label(self.main_frame, width=w*1.5, height=h*1.5)
            # video_label.pack(expand=YES, fill=BOTH)
            #
            # player = tkvideo(processed_video_path, video_label, loop=1)
            # player.play()
        else:
            self.main_frame.destroy()
            print("Video Processed without displaying the result")



FoosGUI(config)


