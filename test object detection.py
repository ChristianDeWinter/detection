import os
import cv2
import torch
import numpy as np
import math
import json
import datetime
import argparse
import subprocess
from ultralytics import YOLO
from for_detect.Inference import LSTM

sport_list = {
    'situp': {
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[14, 12], [15, 13], [6, 12], [7, 13]]
    },
    'pushup': {
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'concerned_key_points_idx': [5, 6, 7, 8, 9, 10],
        'concerned_skeletons_idx': [[9, 11], [7, 9], [6, 8], [8, 10]]
    },
    'squat': {
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'concerned_key_points_idx': [11, 12, 13, 14, 15],
        'concerned_skeletons_idx': [[16, 14], [14, 12], [17, 15], [15, 13]]
    }
}

def calculate_angle(key_points, left_points_idx, right_points_idx):
    def _calculate_angle(line1, line2):
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        return angle_diff

    left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in left_points_idx]
    right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    angle = (angle_left + angle_right) / 2
    return angle

def main():
    # Load push-up count from the file for the current day, if available
    current_date = datetime.datetime.now().strftime("%m/%d/%y")
    pushup_counter = 0
    total_pushup_count = 0
    
    if os.path.exists("pushup_count.txt") and os.stat("pushup_count.txt").st_size != 0:
        with open("pushup_count.txt", "r") as file:
            for line in file:
                # Extract date from the line
                line_parts = line.split(", ")
                if len(line_parts) >= 2:
                    line_date = line_parts[0].split(" ")[1]
                    if current_date == line_date:
                        print("Found entry for today:", line)
                        # Extract push-up count from the line and add it to the total count
                        total_pushup_count += int(line_parts[1].split(" ")[0])

    print("Total push-up count for today:", total_pushup_count)

    model_path = 'model/yolov8s-pose.pt'
    detector_model_path = './for_detect/checkpoint/best_model.pt'
    input_video_path = r'0'

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Load exercise model
    with open(os.path.join(os.path.dirname(detector_model_path), 'idx_2_category.json'), 'r') as f:
        idx_2_category = json.load(f)
    detect_model = LSTM(17*2, 8, 2, 3, model.device)
    model_weight = torch.load(detector_model_path)
    detect_model.load_state_dict(model_weight)

    # Open the video file or camera
    if input_video_path.isnumeric():
        cap = cv2.VideoCapture(int(input_video_path))
    else:
        cap = cv2.VideoCapture(input_video_path)

    # Set variables to record motion status
    reaching = False
    reaching_last = False
    state_keep = False
    prev_angle = None

    # Define thresholds and hysteresis
    maintaining_threshold = sport_list['pushup']['maintaining']
    relaxing_threshold = sport_list['pushup']['relaxing']
    hysteresis = 48.7  # Adjust this value based on experimentation (For Videos = 40 is good and for Camera = 48.7 wroks for skinny jacket/coat dont work (sometimes count Not 100%))

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)

            if results[0].keypoints.shape[1] == 0:
                continue

            angle = calculate_angle(results[0].keypoints, sport_list['pushup']['left_points_idx'], sport_list['pushup']['right_points_idx'])

            if angle < maintaining_threshold - hysteresis:
                reaching = True
            elif angle > relaxing_threshold + hysteresis:
                reaching = False

            if reaching != reaching_last:
                reaching_last = reaching
                if reaching:
                    state_keep = True
                elif not reaching and state_keep:
                    # Check if the angle has crossed the threshold in the opposite direction
                    if prev_angle is not None and prev_angle > relaxing_threshold:
                        pushup_counter += 1
                        total_pushup_count += 1
                    state_keep = False
            prev_angle = angle

            # Set text to display based on push-up count
            if total_pushup_count >= 100:
                text = "Completed daily push-up!s"
            else:
                text = f"Number of Push-ups: {total_pushup_count}/100"
            # Display push-up counter on the video frame
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Push-up Cam", frame)

            # End the program and save push-up count to a text file if 'q' is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                current_time = datetime.datetime.now().strftime("%A %x %I:%M %p")
                with open("pushup_count.txt", "a") as file:
                    file.write(f"{current_time}, {pushup_counter} push-ups\n")
                break
            
            # Check if the push-up count has reached 100
            if pushup_counter == 100:
                print("Challenge Complete!")

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Run the pushup_count_gui.py script after object_detect.py terminates
    subprocess.Popen(["python", "pushup_count_gui.py"])

if __name__ == '__main__':
    main()
