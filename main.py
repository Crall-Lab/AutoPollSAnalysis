import os
import queue
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError as error:
    raise SystemExit(
        "The AutoPollS GUI requires Tk. Install the Tk package for your Python "
        "environment, or use the autopolls-stills command-line runner."
    ) from error

from PIL import Image, ImageTk

from offline_analysis import ap_detector, autopolls_utils


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
MAX_PREVIEW_IMAGES = 500


class apGui:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoPollS Browser")

        self.source_path = tk.StringVar()
        self.csv_path = tk.StringVar()
        self.crop_path = tk.StringVar()
        self.model_path = tk.StringVar(value=os.environ.get("AUTOPOLLS_MODEL_DIR", autopolls_utils.DEFAULT_MODEL_DIR))
        self.write_annotated_videos = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Select a still-image folder or a video, then run detect+classify.")
        self.messages = queue.Queue()
        self.image_files = []
        self.current_image = None
        self.preview_loading = False

        self.build()
        self.root.after(100, self.flush_messages)

    def build(self):
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(root_frame)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.add_folder_row(controls, 0, "Source data", self.source_path, self.browse_source)
        ttk.Button(controls, text="Browse video", command=self.browse_source_video).grid(
            row=0, column=3, sticky="e", padx=(8, 0), pady=3
        )
        self.add_folder_row(controls, 1, "CSV output", self.csv_path, self.browse_csv_output)
        self.add_folder_row(controls, 2, "Crop output", self.crop_path, self.browse_crop_output)
        self.add_folder_row(controls, 3, "Model bundle", self.model_path, self.browse_model)

        ttk.Checkbutton(
            controls,
            text="Write annotated videos",
            variable=self.write_annotated_videos,
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

        self.preview_button = ttk.Button(controls, text="Load preview images", command=self.load_preview_images)
        self.preview_button.grid(row=5, column=1, sticky="e", padx=8, pady=(8, 0))

        self.run_button = ttk.Button(controls, text="Run detect+classify", command=self.ap_analysis)
        self.run_button.grid(row=5, column=2, sticky="e", pady=(8, 0))

        list_frame = ttk.Frame(root_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, width=45, height=20)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self.show_image)

        preview_frame = ttk.Frame(root_frame)
        preview_frame.grid(row=1, column=1, sticky="nsew", padx=(12, 0), pady=(12, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(preview_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.log_box = tk.Text(root_frame, height=9, wrap="word")
        self.log_box.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(12, 0))

        ttk.Label(root_frame, textvariable=self.status).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        root_frame.columnconfigure(0, weight=1)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(1, weight=1)

    def add_folder_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", pady=3)

    def browse_source(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.source_path.set(folder)
        self.image_files = []
        self.listbox.delete(0, tk.END)
        self.image_label.config(image="")
        self.current_image = None
        self.status.set("Source selected. Preview image loading is optional.")

    def browse_source_video(self):
        video = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.avi *.mp4 *.mov *.m4v *.mkv"),
                ("All files", "*.*"),
            ]
        )
        if not video:
            return
        self.source_path.set(video)
        self.image_files = []
        self.listbox.delete(0, tk.END)
        self.image_label.config(image="")
        self.current_image = None
        self.status.set("Source video selected. Annotated videos are optional.")

    def browse_csv_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.csv_path.set(folder)

    def browse_crop_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.crop_path.set(folder)

    def browse_model(self):
        folder = filedialog.askdirectory()
        if folder:
            self.model_path.set(folder)

    def load_preview_images(self):
        folder = self.source_path.get()
        if not folder:
            messagebox.showerror("Missing source", "Select source data first.")
            return
        if os.path.isfile(folder):
            self.status.set("Preview image loading is available for folders, not individual videos.")
            return
        if self.preview_loading:
            return
        self.preview_loading = True
        self.preview_button.config(state="disabled")
        self.status.set("Loading preview images...")
        self.listbox.delete(0, tk.END)
        self.image_files = []
        worker = threading.Thread(target=self.preview_worker, args=(folder,), daemon=True)
        worker.start()

    def preview_worker(self, folder):
        images = []
        truncated = False
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if not filename.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                images.append(os.path.join(root, filename))
                if len(images) >= MAX_PREVIEW_IMAGES:
                    truncated = True
                    self.messages.put(("PREVIEW_DONE", images, truncated))
                    return
        self.messages.put(("PREVIEW_DONE", images, truncated))

    def show_image(self, event):
        if not self.listbox.curselection():
            return
        image_path = self.image_files[self.listbox.curselection()[0]]
        image = Image.open(image_path)
        image.thumbnail((420, 420))
        self.current_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_image)

    def ap_analysis(self):
        required = [
            ("source data", self.source_path.get()),
            ("CSV output", self.csv_path.get()),
            ("crop output", self.crop_path.get()),
            ("model bundle", self.model_path.get()),
        ]
        missing = [label for label, value in required if not value]
        if missing:
            messagebox.showerror("Missing folders", "Select " + ", ".join(missing) + ".")
            return

        self.run_button.config(state="disabled")
        self.log_box.delete("1.0", tk.END)
        self.status.set("Analysis running...")
        worker = threading.Thread(target=self.run_worker, daemon=True)
        worker.start()

    def run_worker(self):
        try:
            runner = ap_detector.intialize(self.model_path.get(), self.messages.put)
            runner.main(
                self.source_path.get(),
                self.csv_path.get(),
                self.crop_path.get(),
                write_annotated_videos=self.write_annotated_videos.get(),
                video_home=self.csv_path.get(),
            )
            self.messages.put("DONE")
        except Exception as error:
            self.messages.put("ERROR: " + str(error))

    def flush_messages(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, tuple) and message[0] == "PREVIEW_DONE":
                images = message[1]
                truncated = message[2]
                self.image_files = images
                self.listbox.delete(0, tk.END)
                source = self.source_path.get()
                for image_path in self.image_files:
                    self.listbox.insert(tk.END, os.path.relpath(image_path, source))
                suffix = " Preview list capped at " + str(MAX_PREVIEW_IMAGES) + "." if truncated else ""
                self.status.set(str(len(self.image_files)) + " preview images loaded." + suffix)
                self.preview_loading = False
                self.preview_button.config(state="normal")
            elif message == "DONE":
                self.status.set("Analysis complete.")
                self.run_button.config(state="normal")
            elif message.startswith("ERROR: "):
                self.status.set("Analysis failed.")
                self.run_button.config(state="normal")
                messagebox.showerror("Analysis failed", message[7:])
            else:
                self.log_box.insert(tk.END, message + "\n")
                self.log_box.see(tk.END)
        self.root.after(100, self.flush_messages)


def main():
    root = tk.Tk()
    apGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
