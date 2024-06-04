import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
import sys
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import calendar

def parse_data_from_file(filename):
    parsed_data = {"dates": [], "push_ups": [], "squats": [], "sit_ups": []}
    with open(filename, 'r') as file:
        for line in file:
            date_str, push_up_str, squats_str, sit_up_str = line.strip().split(", ")
            date = datetime.strptime(date_str, "%A %m/%d/%y %I:%M %p")
            push_ups = int(push_up_str.split()[0])
            squats = int(squats_str.split()[0])
            sit_ups = int(sit_up_str.split()[0])
            parsed_data["dates"].append(date)
            parsed_data["push_ups"].append(push_ups)
            parsed_data["squats"].append(squats)
            parsed_data["sit_ups"].append(sit_ups)
    return parsed_data

def filter_data(parsed_data, interval):
    filtered_dates = []
    filtered_push_ups = []
    filtered_squats = []
    filtered_sit_ups = []

    if interval == "Every Week of the Month":
        current_month = parsed_data["dates"][0].month
        current_year = parsed_data["dates"][0].year

        first_day = datetime(current_year, current_month, 1)
        last_day = datetime(current_year, current_month, calendar.monthrange(current_year, current_month)[1])

        all_mondays = []
        day = first_day
        while day <= last_day:
            if day.weekday() == 0:  # Monday
                all_mondays.append(day)
            day += timedelta(days=1)

        for monday in all_mondays:
            week_start = monday
            week_end = monday + timedelta(days=6)
            week_push_ups = sum(
                parsed_data["push_ups"][i]
                for i, date in enumerate(parsed_data["dates"])
                if week_start <= date <= week_end
            )
            week_squats = sum(
                parsed_data["squats"][i]
                for i, date in enumerate(parsed_data["dates"])
                if week_start <= date <= week_end
            )
            week_sit_ups = sum(
                parsed_data["sit_ups"][i]
                for i, date in enumerate(parsed_data["dates"])
                if week_start <= date <= week_end
            )
            filtered_dates.append(f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
            filtered_push_ups.append(week_push_ups)
            filtered_squats.append(week_squats)
            filtered_sit_ups.append(week_sit_ups)

    elif interval == "Every Month of Year":
        current_year = parsed_data["dates"][0].year
        month_data = {}
        for date, push_ups, squats, sit_ups in zip(parsed_data["dates"], parsed_data["push_ups"], parsed_data["squats"], parsed_data["sit_ups"]):
            if date.year == current_year:
                month = date.month
                if month not in month_data:
                    month_data[month] = {"push_ups": 0, "squats": 0, "sit_ups": 0}
                month_data[month]["push_ups"] += push_ups
                month_data[month]["squats"] += squats
                month_data[month]["sit_ups"] += sit_ups

        for month in range(1, 13):
            if month in month_data:
                filtered_dates.append(calendar.month_name[month])
                filtered_push_ups.append(month_data[month]["push_ups"])
                filtered_squats.append(month_data[month]["squats"])
                filtered_sit_ups.append(month_data[month]["sit_ups"])
            else:
                filtered_dates.append(calendar.month_name[month])
                filtered_push_ups.append(0)
                filtered_squats.append(0)
                filtered_sit_ups.append(0)

    elif interval == "Today":
        today = datetime.now().strftime("%Y-%m-%d")
        today_push_ups = sum(
            parsed_data["push_ups"][i]
            for i, date in enumerate(parsed_data["dates"])
            if date.strftime("%Y-%m-%d") == today
        )
        today_squats = sum(
            parsed_data["squats"][i]
            for i, date in enumerate(parsed_data["dates"])
            if date.strftime("%Y-%m-%d") == today
        )
        today_sit_ups = sum(
            parsed_data["sit_ups"][i]
            for i, date in enumerate(parsed_data["dates"])
            if date.strftime("%Y-%m-%d") == today
        )
        filtered_dates.append(today)
        filtered_push_ups.append(today_push_ups)
        filtered_squats.append(today_squats)
        filtered_sit_ups.append(today_sit_ups)

    return {"dates": filtered_dates, "push_ups": filtered_push_ups, "squats": filtered_squats, "sit_ups": filtered_sit_ups}

def update_chart():
    global parsed_data
    global chart_tab

    selected_interval = filter_var.get()
    
    filtered_data = filter_data(parsed_data, selected_interval)

    for widget in chart_tab.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots()
    ax.bar(filtered_data["dates"], filtered_data["push_ups"], width=0.6, label='Push-ups')
    ax.bar(filtered_data["dates"], filtered_data["squats"], width=0.4, label='Squats')
    ax.bar(filtered_data["dates"], filtered_data["sit_ups"], width=0.4, label='Sit-ups')
    ax.set_xlabel('Date and Time')
    ax.set_ylabel('Number of exercises')
    ax.set_title('Exercises Over Time')
    plt.xticks(rotation=45, ha='right')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=chart_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def run_main_script():
    script_name = "test object detection.py"
    script_path = os.path.join(os.getcwd(), script_name)
    print(f"Running script at path: {script_path}")
    if not os.path.isfile(script_path):
        messagebox.showerror("Error", f"Script file not found: {script_path}")
        return
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running main script: {e}")
    finally:
        global main_script_executed
        main_script_executed = True
        run_button.config(state=tk.DISABLED)
        root.after(0, root.quit)

def run_button_clicked():
    if not main_script_executed:
        threading.Thread(target=run_main_script).start()

def quit_button_clicked():
    if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
        root.quit()
        root.after(0, root.destroy)
        root.after(100, sys.exit)

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

    global filter_var
    filter_var = StringVar(root)
    filter_var.set("Filter Interval")
    filter_dropdown = ttk.Combobox(root, textvariable=filter_var, values=["Today", "Every Week of the Month", "Every Month of Year"])
    filter_dropdown.pack(pady=5)

    filter_button = tk.Button(root, text="Filter", command=update_chart)
    filter_button.pack(pady=5)
    
    global notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    global chart_tab
    chart_tab = ttk.Frame(notebook)
    notebook.add(chart_tab, text='Chart')

    root.mainloop()

if __name__ == "__main__":
    parsed_data = parse_data_from_file("exercise_count.txt")
    create_gui()
