from ultralytics import YOLO
import torch

# Initialize YOLO model
model = YOLO('model/fortnitev8.pt')

# Move the model to CUDA if available
if torch.cuda.is_available():
    print("Cuda Active")
    model = model.to('cuda')
else:
    print("Cuda NOT ACTIVE")

# Perform object detection
result = model(source='video/fortnite.mov', task='detect', show=True, conf=0.2, save=False)

# Wait for user input before exiting
input("Press Enter to exit...")
