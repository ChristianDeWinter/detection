# When test object detection.py is ended by pressing the exit key
import tkinter as tk
from tkinter import messagebox

def show_latest_exercise_counts():
    try:
        with open("exercise_count.txt", "r") as file:
            lines = file.readlines()
            if lines:
                latest_counts = lines[-2:]
                formatted_counts = ""
                
                for count in latest_counts:
                    date_time, exercise_details = count.split(',', 1)
                    formatted_counts += f"{date_time.strip()}\n"
                    exercises = exercise_details.split(',')
                    for exercise in exercises:
                        formatted_counts += f"{exercise.strip()}\n"
                    formatted_counts += "\n"

                messagebox.showinfo("Latest Exercise Counts", formatted_counts.strip())
            else:
                messagebox.showinfo("Exercise Progress", "No exercise data found. Time to hit the floor!")
    except FileNotFoundError:
        messagebox.showerror("Error", "Oops! No exercise data found. Start counting now!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    show_latest_exercise_counts()
    root.destroy()
