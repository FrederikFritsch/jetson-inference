# This script will slice an image and perform object detection using Yolov8.
import time
import ultralytics
from PIL import Image, ImageDraw, ImageFont
import cv2
from sahi.slicing import slice_image
from random import randint

#Variable declarations

IMAGE_PATH = "543.png"
#ANNOTATIONS = ["Airplane", "Vehicle", "Ship"]
#ANNOTATIONS = ["Airplane", "Small Vehicle", "Large Vehicle", "Ship"]
ANNOTATIONS = ["Airplane", "Ship", "Vehicle"]

#MODEL_PATH = "./runs/detect/100_epochs_DIOR/weights/best.pt"
#MODEL_PATH = "./runs/detect/xView_Adam_opt_cont/weights/best.pt"
MODEL_PATH = "bestEXP3.pt"
SLICE_SIZE = 640
SLICE_OVERLAP = 0.2
colors = []
detections = []



#creating colors for the different classes
for i in range(len(ANNOTATIONS)):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

#Initiating the Model
model = ultralytics.YOLO(MODEL_PATH)

#Open image and read size.
original_img = cv2.imread(IMAGE_PATH)
x_size = original_img.shape[1]
y_size = original_img.shape[0]
start_time = time.time()
background_img = Image.open(IMAGE_PATH)

#Slice the image and store results in variable

slice_image_result = slice_image(
    image=IMAGE_PATH,
    slice_height=SLICE_SIZE,
    slice_width=SLICE_SIZE,
    overlap_height_ratio=SLICE_OVERLAP,
    overlap_width_ratio=SLICE_OVERLAP,
)
stop_time = time.time()
print(f"Number of slices: {len(slice_image_result)}")
#Extract bounding box coordinates+identified cls and store in detections
for index, sliced_image in enumerate(slice_image_result.sliced_image_list):
    img = sliced_image.image
    PIL_img = Image.fromarray(img)

    results = model.predict(source=PIL_img, line_thickness=2, verbose=False, max_det=5000, device="cpu", conf=0.5, iou=0.7)

    boxes = results[0].boxes
    x, y = sliced_image.starting_pixel
    for box in boxes:
        cls_index = int(box.cls[0])
        x1 = box.xyxy[0][0]+x
        y1 = box.xyxy[0][1]+y
        x2 = box.xyxy[0][2]+x
        y2 = box.xyxy[0][3]+y
        detections.append({"cls_index":cls_index, "x1":x1, "y1":y1, "x2":x2, "y2":y2})
inference_time = time.time()
print(f"Number of ddetections: {len(detections)}")
print(f"Time for slicing: {stop_time-start_time}")
print(f"Time for inference: {inference_time-stop_time}")
print(f"Time for slicing and inference: {inference_time-start_time}")
#Plot the bounding boxes
draw = ImageDraw.Draw(background_img)
font = ImageFont.load_default()
for detection in detections:
    draw.rectangle([detection["x1"], detection["y1"], detection["x2"], detection["y2"]], outline=colors[detection["cls_index"]-1], width=3)
    cls_index = int(detection["cls_index"])
    cls_text = ANNOTATIONS[cls_index]
    draw.text((detection["x1"], detection["y1"]-10), cls_text, font=font, fill=(255,255,255,255))


#background_img.save("Results_Image.png")
background_img.show()