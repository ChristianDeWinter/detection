import subprocess
import tkinter as tk
from tkinter import messagebox
import sys
import threading

def run_main_script():
    try:
        subprocess.run(["python", "test object detection.py"], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running main script: {e}")
    finally:
        global main_script_executed
        main_script_executed = True
        run_button.config(state=tk.DISABLED)
        root.quit()
        root.destroy() 

def run_button_clicked():
    if not main_script_executed:
 
        threading.Thread(target=run_main_script).start()

def quit_button_clicked():
    if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
        root.destroy()

def create_gui():
    global root
    root = tk.Tk()
    root.title("Run Main Script")
    global main_script_executed
    main_script_executed = False

    label = tk.Label(root, text="Click the button to run the main script.")
    label.pack(pady=10)

    global run_button
    run_button = tk.Button(root, text="Run", command=run_button_clicked)
    run_button.pack(pady=5)

    quit_button = tk.Button(root, text="Quit", command=quit_button_clicked)
    quit_button.pack(pady=5)

    root.mainloop()
    
if __name__ == "__main__":
    create_gui()
