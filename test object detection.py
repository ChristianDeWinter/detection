import os
import cv2
import torch
import numpy as np
import math
import json
import datetime
import argparse
import subprocess
import pygame
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
        'concerned_key_points_idx': [11, 12, 13, 14, 15, 16],
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
        left_points[1][0], left_points[1][1],
        left_points[0][0], left_points[0][1]
    ]
    line2_left = [
        left_points[1][0], left_points[1][1],
        left_points[2][0], left_points[2][1]
    ]
    angle_left = _calculate_angle(line1_left, line2_left)

    line1_right = [
        right_points[1][0], right_points[1][1],
        right_points[0][0], right_points[0][1]
    ]
    line2_right = [
        right_points[1][0], right_points[1][1],
        right_points[2][0], right_points[2][1]
    ]
    angle_right = _calculate_angle(line1_right, line2_right)

    angle = (angle_left + angle_right) / 2
    return angle

def main():
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load push-up, squat, and sit-up counts from the file for the current day, if available
    current_date = datetime.datetime.now().strftime("%m/%d/%y")
    pushup_counter = 0
    squat_counter = 0
    situp_counter = 0
    total_pushup_count = 0
    total_squat_count = 0
    total_situp_count = 0
    daily_pushup_goal = 100
    daily_squat_goal = 100
    daily_situp_goal = 100

    if os.path.exists("exercise_count.txt") and os.stat("exercise_count.txt").st_size != 0:
        with open("exercise_count.txt", "r") as file:
            for line in file:
                line_parts = line.split(", ")
                if len(line_parts) >= 4:
                    line_date = line_parts[0].split(" ")[1]
                    if current_date == line_date:
                        if "push-ups" in line_parts[1]:
                            total_pushup_count += int(line_parts[1].split(" ")[0])
                        if "squats" in line_parts[2]:
                            total_squat_count += int(line_parts[2].split(" ")[0])
                        if "sit-ups" in line_parts[3]:
                            total_situp_count += int(line_parts[3].split(" ")[0])

    print("Total push-up count for today:", total_pushup_count)
    print("Total squat count for today:", total_squat_count)
    print("Total sit-up count for today:", total_situp_count)

    model_path = 'model/yolov8s-pose.engine'
    detector_model_path = './for_detect/checkpoint/best_model.pt'
    input_video_path = r'C:\Users\CTRL C and CTRL V\Documents\bitacademy\Project\motivation software app\detection\video\pushup2.mp4'
    exit_key = "q"
    model = YOLO(model_path)

    with open(os.path.join(os.path.dirname(detector_model_path), 'idx_2_category.json'), 'r') as f:
        idx_2_category = json.load(f)
    detect_model = LSTM(17*2, 8, 2, 3, model.device)
    model_weight = torch.load(detector_model_path)
    detect_model.load_state_dict(model_weight)

    if input_video_path.isnumeric():
        cap = cv2.VideoCapture(int(input_video_path))
    else:
        cap = cv2.VideoCapture(input_video_path)

    reaching_pushup = False
    reaching_squat = False
    reaching_situp = False
    reaching_last_pushup = False
    reaching_last_squat = False
    reaching_last_situp = False
    state_keep_pushup = False
    state_keep_squat = False
    state_keep_situp = False
    prev_angle_pushup = None
    prev_angle_squat = None
    prev_angle_situp = None

    maintaining_threshold_pushup = sport_list['pushup']['maintaining']
    relaxing_threshold_pushup = sport_list['pushup']['relaxing']
    maintaining_threshold_squat = sport_list['squat']['maintaining']
    relaxing_threshold_squat = sport_list['squat']['relaxing']
    maintaining_threshold_situp = sport_list['situp']['maintaining']
    relaxing_threshold_situp = sport_list['situp']['relaxing']
    hysteresis = 48.7

    pushup_sound = pygame.mixer.Sound(r'C:\Users\CTRL C and CTRL V\Documents\bitacademy\Project\motivation software app\detection\sound\ding.mp3')
    squat_sound = pygame.mixer.Sound(r'C:\Users\CTRL C and CTRL V\Documents\bitacademy\Project\motivation software app\detection\sound\ding.mp3')
    situp_sound = pygame.mixer.Sound(r'C:\Users\CTRL C and CTRL V\Documents\bitacademy\Project\motivation software app\detection\sound\ding.mp3')

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)

            if results[0].keypoints.shape[1] == 0:
                continue

            angle_pushup = calculate_angle(results[0].keypoints, sport_list['pushup']['left_points_idx'], sport_list['pushup']['right_points_idx'])
            angle_squat = calculate_angle(results[0].keypoints, sport_list['squat']['left_points_idx'], sport_list['squat']['right_points_idx'])
            angle_situp = calculate_angle(results[0].keypoints, sport_list['situp']['left_points_idx'], sport_list['situp']['right_points_idx'])

            if angle_pushup < maintaining_threshold_pushup - hysteresis:
                reaching_pushup = True
            elif angle_pushup > relaxing_threshold_pushup + hysteresis:
                reaching_pushup = False

            if reaching_pushup != reaching_last_pushup:
                reaching_last_pushup = reaching_pushup
                if reaching_pushup:
                    state_keep_pushup = True
                elif not reaching_pushup and state_keep_pushup:
                    if prev_angle_pushup is not None and prev_angle_pushup > relaxing_threshold_pushup:
                        pushup_counter += 1
                        total_pushup_count += 1
                        pushup_sound.play()
                    state_keep_pushup = False
            prev_angle_pushup = angle_pushup

            if angle_squat < maintaining_threshold_squat - hysteresis:
                reaching_squat = True
            elif angle_squat > relaxing_threshold_squat + hysteresis:
                reaching_squat = False

            if reaching_squat != reaching_last_squat:
                reaching_last_squat = reaching_squat
                if reaching_squat:
                    state_keep_squat = True
                elif not reaching_squat and state_keep_squat:
                    if prev_angle_squat is not None and prev_angle_squat > relaxing_threshold_squat:
                        squat_counter += 1
                        total_squat_count += 1
                        squat_sound.play()
                    state_keep_squat = False
            prev_angle_squat = angle_squat
            
            if angle_situp < maintaining_threshold_situp - hysteresis:
                reaching_situp = True
            elif angle_situp > relaxing_threshold_situp + hysteresis:  # Fixed the typo here
                reaching_situp = False

            if reaching_situp != reaching_last_situp:
                reaching_last_situp = reaching_situp
                if reaching_situp:
                    state_keep_situp = True
                elif not reaching_situp and state_keep_situp:
                    if prev_angle_situp is not None and prev_angle_situp > relaxing_threshold_situp:
                        situp_counter += 1
                        total_situp_count += 1
                        situp_sound.play()
                    state_keep_situp = False
            prev_angle_situp = angle_situp

            text_lines = []
            if total_pushup_count >= daily_pushup_goal:
                text_lines.append(f"Push-ups: {total_pushup_count}/{daily_pushup_goal} - Completed!")
            else:
                text_lines.append(f"Push-ups: {total_pushup_count}/{daily_pushup_goal}")

            if total_squat_count >= daily_squat_goal:
                text_lines.append(f"Squats: {total_squat_count}/{daily_squat_goal} - Completed!")
            else:
                text_lines.append(f"Squats: {total_squat_count}/{daily_squat_goal}")

            if total_situp_count >= daily_situp_goal:
                text_lines.append(f"Sit-ups: {total_situp_count}/{daily_situp_goal} - Completed!")
            else:
                text_lines.append(f"Sit-ups: {total_situp_count}/{daily_situp_goal}")

            text2 = f"Press: {exit_key} to quit"

            for i, line in enumerate(text_lines):
                cv2.putText(frame, line, (20, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text2, (20, 50 + len(text_lines) * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Exercise Cam", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord(f"{exit_key}"):
                current_time = datetime.datetime.now().strftime("%A %x %I:%M %p")
                with open("exercise_count.txt", "a") as file:
                    file.write(f"{current_time}, {pushup_counter} push-ups, {squat_counter} squats, {situp_counter} sit-ups\n")
                break
            
            if total_pushup_count >= daily_pushup_goal and total_squat_count >= daily_squat_goal and total_situp_count >= daily_situp_goal:
                print("Challenge Complete!")
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    subprocess.Popen(["python", "exercise_count_message.py"])

if __name__ == '__main__':
    main()
