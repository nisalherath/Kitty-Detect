import os
import torch
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import threading
import webbrowser
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import tkinter as tk

from yolo_cat_detect import download_yolo_model, load_yolo_model, detect_cat_with_yolo


# Initialize paths
MODEL_PATH = r'models/cat_classifier.pth'
YOLO_MODEL_PATH = r'models/yolov5s.pt'

# Download YOLO model if it does not exist
download_yolo_model()

# Initialize the ResNet50 model architecture
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes (my_cat and other_cats)

# Load the Custom Trained model of mine
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True), strict=False)
model.eval()

# Load YOLO model
yolo_model = load_yolo_model()

# Input image Compared Trough YoLo
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item()


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if not file_path:
        return

    img = Image.open(file_path)

    # Scale image to 250x250
    max_size = 250
    img.thumbnail((max_size, max_size))

    # Fake Bounding Box
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, img.width - 10, img.height - 10], outline="red", width=1)

    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk

    label_result.config(text="Classifying...", fg="#FF6600")  # Darker orange
    threading.Thread(target=lambda: classify_and_display(file_path)).start()


def classify_and_display(file_path):

    is_cat = detect_cat_with_yolo(file_path, yolo_model)

    if not is_cat:
        label_result.config(text="Error: Not a cat image!", fg="red")
        return

    predicted, confidence = classify_image(file_path)

    root.after(100, lambda: animate_confidence_circle(confidence * 100))

    if predicted == 0:
        result_text = f"This is My Kitty\n{confidence * 100:.2f}% Confidence"
        label_result.config(fg="green")
        target_color = "green"
    else:
        result_text = f"Who is this? Not My Kitty!\n{confidence * 100:.2f}% Confidence"
        label_result.config(fg="red")
        target_color = "red"

    label_result.config(text=result_text)

    root.after(100, lambda: update_confidence_circle_color(confidence * 100, target_color))



def animate_confidence_circle(target_confidence):
    current_confidence = 0
    step = 2

    def update_circle():
        nonlocal current_confidence
        if current_confidence < target_confidence:
            current_confidence += step
            update_confidence_circle(current_confidence)
            root.after(10, update_circle)

    update_circle()


def update_confidence_circle(confidence):
    canvas.delete("confidence")

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if canvas_width == 1 or canvas_height == 1:
        root.after(100, lambda: update_confidence_circle(confidence))
        return

    x, y = canvas_width // 2, canvas_height // 2
    r = min(canvas_width, canvas_height) // 3  # radius
    start_angle = 90
    extent = -3.59 * confidence  # % to degrees

    # background circle
    canvas.create_oval(x - r, y - r, x + r, y + r, outline="#e0e0e0", width=10, tags="confidence")

    # confidence arc (draw a single arc with the desired extent)
    canvas.create_arc(
        x - r, y - r, x + r, y + r,
        start=start_angle, extent=extent,
        outline=get_gradient_color(confidence / 100), width=10, style=tk.ARC, tags="confidence"
    )

    # percentage text (circle)
    canvas.create_text(x, y, text=f"{confidence:.1f}%", font=("Poppins", 12, "bold"), fill="white", tags="confidence")


def update_confidence_circle_color(confidence, target_color):
    canvas.delete("confidence")

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if canvas_width == 1 or canvas_height == 1:
        root.after(100, lambda: update_confidence_circle_color(confidence, target_color))
        return

    x, y = canvas_width // 2, canvas_height // 2
    r = min(canvas_width, canvas_height) // 3  # radius
    start_angle = 90
    extent = -3.59 * confidence  # % to degrees

    # background circle
    canvas.create_oval(x - r, y - r, x + r, y + r, outline="#e0e0e0", width=5, tags="confidence")

    canvas.create_arc(
        x - r, y - r, x + r, y + r,
        start=start_angle, extent=extent,
        outline=target_color, width=5, style=tk.ARC, tags="confidence"
    )

    # percentage text (circle)
    canvas.create_text(x, y, text=f"{confidence:.1f}%", font=("Poppins", 12, "bold"), fill="white", tags="confidence")


def get_gradient_color(progress):
    if progress < 0.5:
        r = int(255 * (progress * 2))
        g = 255
        b = 0
    else:
        r = 255
        g = int(255 * ((1 - progress) * 2))
        b = 0

    return f"#{r:02x}{g:02x}{b:02x}"


# Open GitHub
def open_github(event):
    webbrowser.open("https://github.com/nisalherath")


# Main tkinter window
root = tk.Tk()
root.title("isdatmyKitty")  # AppName
root.geometry("700x700")  # App's Initial Window
root.resizable(True, True)
root.config(padx=20, pady=20, bg="black")

frame = tk.Frame(root, bg="black")
frame.pack(fill=tk.BOTH, expand=True)


def on_enter(e):
    button_open.config(bg="#FF6600")  # Darker orange
    button_open.config(cursor="hand2")


def on_leave(e):
    button_open.config(bg="#FF4500")  # Darker orange
    button_open.config(cursor="")


button_open = tk.Button(frame, text="Open Image", command=open_file, font=("Poppins", 12), bg="#FF4500", fg="white",
                        relief="raised", padx=10, pady=5, bd=0, highlightthickness=0)
button_open.pack(pady=10)
button_open.bind("<Enter>", on_enter)
button_open.bind("<Leave>", on_leave)

# Image Label
label_image = tk.Label(frame, bg="black")
label_image.pack(pady=10)

# Result Label
label_result = tk.Label(frame, text="Result will appear here", font=("Poppins", 16), bg="black", fg="white",
                        wraplength=500)
label_result.pack(pady=10, expand=True)

# Confidence Circle Canvas
canvas = tk.Canvas(frame, bg="black", highlightthickness=0)
canvas.pack(pady=10, expand=True)

# Footer Content
footer_text = "isdatmyKitty Version 1.0 - Created by Nisal Herath."

# Footer Label
footer = tk.Label(root, text=footer_text, font=("Poppins", 10), bg="black", fg="gray")
footer.pack(side="bottom", fill=tk.X, pady=10)


def on_enter_github(e):
    footer_link.config(fg="#FF6600")
    footer_link.config(cursor="hand2")


def on_leave_github(e):
    footer_link.config(fg="white")
    footer_link.config(cursor="")


footer_link = tk.Label(root, text="My GitHub", font=("Poppins", 10, "underline"), fg="white", bg="black")
footer_link.pack(side="bottom", pady=5)

# Hover events and the click event
footer_link.bind("<Button-1>", open_github)
footer_link.bind("<Enter>", on_enter_github)
footer_link.bind("<Leave>", on_leave_github)

# Start the GUI
root.mainloop()
