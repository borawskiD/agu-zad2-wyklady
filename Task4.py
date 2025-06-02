import os
import yaml
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

READ_MODEL = False   # True -> cache, False -> trening
MODEL_PATH = "models/calculations/weights/best.pt"
DATA_YAML_PATH = "dataset/audio.v3i.yolov8-obb/data.yaml"

with open(DATA_YAML_PATH) as f:
    data_config = yaml.safe_load(f)

TRAIN_DIR = data_config["train"]
VAL_DIR = data_config["val"]
TEST_DIR = data_config.get("test", VAL_DIR)

NC = data_config["nc"]
CLASS_NAMES = data_config["names"]

if READ_MODEL and os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = YOLO("yolov8n.pt")
    model.train(data=DATA_YAML_PATH, epochs=1, imgsz=640, project="models", name="calculations", exist_ok=True)
    model = YOLO("models/calculations/weights/best.pt")

def load_labels(path_txt):
    boxes = []
    if not os.path.exists(path_txt):
        return boxes
    with open(path_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if len(parts) >= 9:
                pts = list(map(float, parts[1:9]))
                boxes.append((cls, pts))
            else:
                cx, cy, w, h = map(float, parts[1:5])
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes.append((cls, [x1, y1, x2, y1, x2, y2, x1, y2]))
    return boxes

def plot_boxes(img, boxes, color=(0, 255, 0), label_prefix='GT'):
    for (cls, pts) in boxes:
        pts_int = [(int(pts[i] * img.shape[1]), int(pts[i + 1] * img.shape[0])) for i in range(0, 8, 2)]
        pts_np = np.array(pts_int, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=2)
        cv2.putText(img, f"{label_prefix}:{CLASS_NAMES[cls]}", pts_int[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def perspective_transform(img, src_points, size=(640, 640)):
    dst_points = np.array([[0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1], [0, size[1]-1]], dtype=np.float32)
    src_points = np.array(src_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, matrix, size)
    return warped

test_image_dir = os.path.join("dataset/audio.v3i.yolov8-obb/test/images")
img_files = [f for f in os.listdir(test_image_dir) if f.endswith((".jpg", ".png"))][:5]

for img_name in img_files:
    img_path = os.path.join(test_image_dir, img_name)
    txt_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_boxes = load_labels(txt_path)
    img_gt = img_rgb.copy()
    plot_boxes(img_gt, gt_boxes, color=(0, 255, 0), label_prefix="GT")

    results = model.predict(img_path, imgsz=640, conf=0.25)
    pred_boxes = []
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = xyxy
            pred_boxes.append((int(box.cls[0]), [
                x1 / img.shape[1], y1 / img.shape[0],
                x2 / img.shape[1], y1 / img.shape[0],
                x2 / img.shape[1], y2 / img.shape[0],
                x1 / img.shape[1], y2 / img.shape[0],
            ]))

    img_pred = img_rgb.copy()
    plot_boxes(img_pred, pred_boxes, color=(255, 0, 0), label_prefix="P")

    graph_candidates = [b for b in pred_boxes if CLASS_NAMES[b[0]] == "chart"]
    if graph_candidates:
        graph_box = graph_candidates[0]
        pts = [(graph_box[1][i] * img.shape[1], graph_box[1][i+1] * img.shape[0]) for i in range(0, 8, 2)]
        warped = perspective_transform(img_rgb, pts, size=(640, 640))
    else:
        warped = img_rgb.copy()

    FREQS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    GRID_WIDTH = warped.shape[1]
    GRID_HEIGHT = warped.shape[0]

    def pixel_to_freq(x_px):
        freq_step = GRID_WIDTH / (len(FREQS) - 1)
        idx = int(round(x_px / freq_step))
        idx = max(0, min(len(FREQS)-1, idx))
        return FREQS[idx]

    def pixel_to_db(y_px):
        return round((y_px / GRID_HEIGHT) * 120.0, 1)

    point_classes = {
        "mark_left": "LEFT",
        "mark_right": "RIGHT"
    }

    audiogram_data = {"LEFT": {}, "RIGHT": {}}

    for cls_id, pts in pred_boxes:
        cls_name = CLASS_NAMES[cls_id]
        if cls_name in point_classes:
            symbol = point_classes[cls_name]
            x_px = int((pts[0] + pts[4]) / 2 * warped.shape[1])
            y_px = int((pts[1] + pts[5]) / 2 * warped.shape[0])
            freq = pixel_to_freq(x_px)
            db_val = pixel_to_db(y_px)
            audiogram_data[symbol][freq] = db_val

    print("FREQ:   ", "  ".join(f"{f:<5}" for f in FREQS))
    for ear in ["LEFT", "RIGHT"]:
        line = [str(audiogram_data[ear].get(f, "-")) for f in FREQS]
        print(f"{ear:<7} {'  '.join(line)}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Oryginał")
    axs[0].axis("off")
    axs[1].imshow(img_pred)
    axs[1].set_title("Predykcja (YOLO)")
    axs[1].axis("off")
    axs[2].imshow(warped)
    axs[2].set_title("Siatka po przekształceniu")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()
