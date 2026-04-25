import torch
import cv2
import numpy as np
import time
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights
)

# -----------------------------
# Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load Model
# -----------------------------
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.to(device)
model.eval()

# -----------------------------
# Faster Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),   # reduced from 512 → faster
    transforms.ToTensor(),
])

# -----------------------------
# Color Map
# -----------------------------
def decode_segmap(label_mask):
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128],
    ], dtype=np.uint8)
    return colors[label_mask % len(colors)]

# -----------------------------
# Load Video
# -----------------------------
cap = cv2.VideoCapture("videos/test.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
input_fps = cap.get(cv2.CAP_PROP_FPS)
print("Input FPS:", input_fps)

out = cv2.VideoWriter("outputs/output.mp4", fourcc, input_fps, (width, height))

# -----------------------------
# Processing Loop
# -----------------------------
frame_count = 0
start_time = time.time()

skip_rate = 3
last_overlay = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔥 Only run model every N frames
    if frame_count % skip_rate == 0 or last_overlay is None:

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)["out"][0]

        output_predictions = output.argmax(0).cpu().numpy()

        # Decode segmentation
        colored_mask = decode_segmap(output_predictions)

        # Resize to original frame
        colored_mask = cv2.resize(
            colored_mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Overlay
        overlay = (0.6 * image + 0.4 * colored_mask).astype(np.uint8)

        # Highlight cars
        car_mask = (output_predictions == 7).astype(np.uint8)
        car_mask = cv2.resize(
            car_mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        overlay[car_mask == 1] = [255, 0, 0]

        last_overlay = overlay

    else:
        # Reuse previous result
        overlay = last_overlay

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # FPS Counter
    # -----------------------------
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # cv2.putText(
    #    overlay_bgr,
    #   f"FPS: {fps:.2f}",
    #   (20, 40),
    #   cv2.FONT_HERSHEY_SIMPLEX,
    #   1,
    #   (0, 255, 0),
    #   2
    # )

    # Write + Display
    out.write(overlay_bgr)
    cv2.imshow("Semantic Segmentation", overlay_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()