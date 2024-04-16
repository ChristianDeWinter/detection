import tkinter as tk
from tkinter import messagebox

def show_results():
    try:
        with open("pushup_count.txt", "r") as file:
            lines = file.readlines()
            if lines:
                latest_result = lines[-1]  # Get the last line
                messagebox.showinfo("Latest Push-up Count", latest_result)
            else:
                messagebox.showinfo("Push-up Count Results", "No push-up count data found.")
    except FileNotFoundError:
        messagebox.showerror("Error", "No push-up count data found.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    show_results()
    root.destroy()
