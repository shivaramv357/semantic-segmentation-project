# Semantic Scene Understanding for Autonomous Driving (Video-Based)

This project demonstrates a perception pipeline for autonomous driving using deep learning-based semantic segmentation.

---

## 🚀 Features
- Semantic segmentation using DeepLabV3 (PyTorch)
- Video-based frame-by-frame processing
- Overlay visualization of segmentation output
- Car highlighting using class-based masking

---

## 🛠️ Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy

---

## ▶️ How to Run

1. Clone the repository:

         git clone https://github.com/shivaramv357/semantic-segmentation-project.git
      
         cd semantic-segmentation-project

3. Install dependencies:

          pip install -r requirements.txt

5. Run the project:

         python main.py

---

## 📸 Output
### 🎥 Output Video
[Click to watch video](outputs/output.mp4)

---

## ⚙️ Pipeline

  1. Input video is processed frame-by-frame
  2. Frames are passed to DeepLabV3 model
  3. Model performs pixel-wise classification
  4. Segmentation output is overlaid on original frames

---

## 📌 Notes
 - Uses pretrained DeepLabV3 model
 - DeepLabV3 uses convolutional neural networks for pixel-wise classification
 - Implemented frame skipping to improve inference speed while maintaining visual consistency

---

## Limitations
- Uses pretrained model without fine-tuning
- Performance may vary under different lighting/weather conditions
