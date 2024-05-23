#When test object detection.py is ended by pressing the exit key
import tkinter as tk
from tkinter import messagebox

def show_latest_pushup_count():
    try:
        with open("pushup_count.txt", "r") as file:
            lines = file.readlines()
            if lines:
                latest_count = lines[-1]
                messagebox.showinfo("Latest Push-up Count", f"You've conquered {latest_count}")
            else:
                messagebox.showinfo("Push-up Progress", "No push-up data found. Time to hit the floor!")
    except FileNotFoundError:
        messagebox.showerror("Error", "Oops! No push-up data found. Start counting now!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    show_latest_pushup_count()
    root.destroy()
