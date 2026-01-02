# dashboard.py
import customtkinter as ctk
from PIL import Image, ImageTk
import json
import os
import tkintermapview

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SHARED_FILE = "latest_detection.json"

class Dashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("POTHOLE / PERSON DETECTOR")
        self.geometry("1300x700")

        left = ctk.CTkFrame(self, width=640, height=480)
        left.pack(side="left", padx=20, pady=20)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="LIVE DETECTION", font=("Arial", 24, "bold")).pack(pady=10)
        self.img_label = ctk.CTkLabel(left, text="No detection yet...", font=("Arial", 18))
        self.img_label.pack(expand=True)

        right = ctk.CTkFrame(self)
        right.pack(side="right", padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(right, text="LIVE LOCATION MAP", font=("Arial", 24, "bold")).pack(pady=10)

        self.map = tkintermapview.TkinterMapView(right, width=600, height=400, corner_radius=15)
        self.map.pack(pady=10)
        self.map.set_position(20.0, 78.0)
        self.map.set_zoom(18)

        self.status = ctk.CTkLabel(right, text="Waiting...", font=("Arial", 16), text_color="orange")
        self.status.pack(pady=5)
        self.gps = ctk.CTkLabel(right, text="GPS: --", font=("Arial", 16))
        self.gps.pack(pady=2)
        self.conf = ctk.CTkLabel(right, text="Confidence: --", font=("Arial", 16))
        self.conf.pack(pady=2)

        self.previous_marker = None
        self.check_file()

    def check_file(self):
        if os.path.exists(SHARED_FILE):
            try:
                with open(SHARED_FILE) as f:
                    data = json.load(f)

                img = Image.open(data["image"]).resize((620, 440))
                photo = ImageTk.PhotoImage(img)
                self.img_label.configure(image=photo, text="")
                self.img_label.image = photo

                lat = data["lat"]
                lon = data["lon"]

                if self.previous_marker:
                    self.previous_marker.delete()

                self.previous_marker = self.map.set_marker(lat, lon, text=f"Conf: {data['conf']}")

                self.map.set_position(lat, lon)

                self.status.configure(text="LIVE DETECTING!", text_color="#00ff88")
                self.gps.configure(text=f"GPS: {lat:.6f}, {lon:.6f}")
                self.conf.configure(text=f"Confidence: {data['conf']:.2%}")

            except Exception as e:
                print(e)

        self.after(1000, self.check_file)

if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()