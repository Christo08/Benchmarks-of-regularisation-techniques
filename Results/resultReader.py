import math
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import json
import seaborn as sns
from tabulate import tabulate
import scipy.stats as stats

plt.rcParams.update({'font.size': 12})

def load_files():
    global datasets
    file_paths = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])

    if file_paths:
        try:
            datasetNames = []
            datasets = []
            for file_path in file_paths:
                # Load the dataset from the json file
                with open(file_path, 'r') as file:
                    dataset = json.load(file)

                datasets.append(dataset)
                # Update the combo box with the column names
                datasetNames.append(dataset["dataset_name"])
                dataset_combo['values'] = datasetNames
                dataset_combo['state'] = 'normal'  # Set default value
        except pd.errors.EmptyDataError:
            print("Selected file is empty.")
        except pd.errors.ParserError:
            print("Error parsing the selected file. Please make sure it's a valid CSV file.")
    else:
        print("No file selected.")


def find_and_checked_checkbox(name):
    counter = 0
    for box in regularisation_method_checkboxes:
        if box.cget('text') == name and regularisation_method_checkboxes_values[counter].get():
            return True
        counter += 1
    return False


def load_dataset(event):
    global dataset, datasets, regularisation_method_checkboxes_values, regularisation_method_checkboxes, lastRow

    try:
        for object in datasets:
            if object['dataset_name'] == dataset_combo.get():
                dataset = object
                break

        value_type = ["accuracies", "losses", "f1_scores"]
        metric_type_combo['values'] = value_type
        metric_type_combo['state'] = 'normal'

        value = ["differences in training and testing", "differences in training and validation", "testing", "training",
                 "validation"]
        set_type_combo['values'] = value
        set_type_combo['state'] = 'normal'
        chart_type_combo['state'] = 'normal'

        for box in regularisation_method_checkboxes:
            box.destroy()
        regularisation_method_checkboxes.clear()
        regularisation_methods = []
        regularisation_method_checkboxes_values = []
        counter = 0
        for run in dataset['runs']:
            if not (run['method'] in regularisation_methods):
                regularisation_method_checkboxes_values.append(tk.BooleanVar(value=True))
                regularisation_methods.append(run["method"])
                regularisation_method_checkbox = ttk.Checkbutton(window, text=run["method"],
                                                                 variable=regularisation_method_checkboxes_values[
                                                                     counter])
                regularisation_method_checkbox.grid(row=math.floor(counter / 5) + 1, column=counter % 5, pady=10)
                lastRow = math.floor(counter / 5) + 1
                regularisation_method_checkboxes.append(regularisation_method_checkbox)
                counter += 1

        # Enable the "Plot Line Chart" button
        plot_button['state'] = 'normal'
        if chart_type_combo.get() == "Box and whisker":
            show_avg_and_std_button['state'] = 'normal'
            show_test_button['state'] = 'normal'
        else:
            show_avg_and_std_button['state'] = 'disabled'
            show_test_button['state'] = 'disabled'

    except pd.errors.EmptyDataError:
        print("Selected file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the selected file. Please make sure it's a valid CSV file.")


def plot_line_chart():
    global dataset, lastRow, canvas
    selected_column = dataset_combo.get()
    yValue = []
    labels = []

    # Create a figure and axis
    if set_type_combo.get() == "differences in training and testing":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                first = True
                for fold in zip(run["results"][metric_type_combo.get()]["training"],
                                run["results"][metric_type_combo.get()]["testing"]):
                    if first:
                        yValue.append([x - y for x, y in zip(fold[0], fold[1])])
                        first = False
                    else:
                        yValue[-1] = [x + y for x, y in zip(yValue[-1], [a - b for a, b in zip(fold[0], fold[1])])]
                yValue[-1] = [x / len(run["results"][metric_type_combo.get()]["training"]) for x in yValue[-1]]
                labels.append(run['method'])
    elif set_type_combo.get() == "differences in training and validation":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                first = True
                for fold in zip(run["results"][metric_type_combo.get()]["training"],
                                run["results"][metric_type_combo.get()]["validation"]):
                    if first:
                        yValue.append([x - y for x, y in zip(fold[0], fold[1])])
                        first = False
                    else:
                        yValue[-1] = [x + y for x, y in zip(yValue[-1], [a - b for a, b in zip(fold[0], fold[1])])]
                yValue[-1] = [x / len(run["results"][metric_type_combo.get()]["training"]) for x in yValue[-1]]
                labels.append(run['method'])
    else:
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                first = True
                for fold in run["results"][metric_type_combo.get()][set_type_combo.get()]:
                    if first:
                        yValue.append(fold)
                        first = False
                    else:
                        yValue[-1] = [x + y for x, y in zip(yValue[-1], fold)]
                yValue[-1] = [x / len(run["results"][metric_type_combo.get()][set_type_combo.get()]) for x in
                              yValue[-1]]
                labels.append(run['method'])

    fig, ax = plt.subplots(figsize=(15, 8))
    epochs = [i + 1 for i in range(len(yValue[0]))]
    for index in range(len(yValue)):
        ax.plot(epochs, yValue[index], label=labels[index])

    # Set labels and title
    ax.set_xlabel('Epochs', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric_type_combo.get().replace("_", " ").capitalize(), fontweight='bold', fontsize=12)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    # Display the plot on the canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=lastRow + 1, column=0, columnspan=6)


def plot_box_chart():
    global dataset, lastRow, canvas
    xValue = {}

    if set_type_combo.get() == "differences in training and testing":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                value = [x[-1] - y[-1] for x, y in zip(run["results"][metric_type_combo.get()]["training"],
                                                       run["results"][metric_type_combo.get()]["testing"])]
                xValue[run['method']] = value
    elif set_type_combo.get() == "differences in training and validation":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                value = [x[-1] - y[-1] for x, y in zip(run["results"][metric_type_combo.get()]["training"],
                                                       run["results"][metric_type_combo.get()]["validation"])]
                xValue[run['method']] = value
    else:
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                method = run['method'].replace(" ", "\n")
                xValue[method] = []
                for fold in run["results"][metric_type_combo.get()][set_type_combo.get()]:
                    xValue[method].append(fold[-1])

    title = metric_type_combo.get().replace("_", " ")
    if metric_type_combo.get() == "f1_scores":
        baseline_mean = np.mean(xValue["Baseline"], axis=0)
        for run in dataset['runs']:
            method = run['method'].replace(" ", "\n")
            if find_and_checked_checkbox(run['method']):
                xValue[method] = xValue[method] - baseline_mean
        title = "Normalized " + title

    fig, ax = plt.subplots(figsize=(15, 8))

    sns.boxplot(data=xValue, orient='v', ax=ax)
    ax.set_ylabel(title, fontweight='bold', fontsize=12)
    ax.set_xlabel("Techniques", fontweight='bold', fontsize=12)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=lastRow + 1, column=0, columnspan=6)


def save_charts():
    folder_path = filedialog.askdirectory()
    figName = dataset_combo.get()+".svg"
    plt.savefig(folder_path + "/" + figName, bbox_inches='tight', pad_inches=0.25, format="svg")


def plot_charts():
    global canvas, save_button
    if canvas is not None:
        canvas.get_tk_widget().delete('All')
    if chart_type_combo.get() == "Box and whisker":
        plot_box_chart()
    else:
        plot_line_chart()
    save_button['state'] = 'normal'


def create_table():
    global dataset
    values = []
    names = []

    if set_type_combo.get() == "differences in training and testing":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                values.append([x[-1] - y[-1] for x, y in zip(run["results"][metric_type_combo.get()]["training"],
                                                             run["results"][metric_type_combo.get()]["testing"])])
                names.append(run["method"])
    elif set_type_combo.get() == "differences in training and validation":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                values.append([x[-1] - y[-1] for x, y in zip(run["results"][metric_type_combo.get()]["training"],
                                                             run["results"][metric_type_combo.get()]["validation"])])
                names.append(run["method"])
    else:
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                values.append([])
                for fold in run["results"][metric_type_combo.get()][set_type_combo.get()]:
                    values[-1].append(fold[-1])
                names.append(run["method"])

    print("Dataset " + dataset_combo.get())
    data = [
        ["Name", "Avg +- Std"]
    ]

    for count, name in enumerate(names):
        avg = round(np.mean(values[count]), 3)
        std = round(np.std(values[count]), 3)
        data.append([name, str(avg) + "+-" + str(std)])

    # Display the table
    print(tabulate(data, headers="firstrow"))


def create_test():
    global dataset
    values = []
    names = []
    combos = []

    if set_type_combo.get() == "differences in training and testing":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                values.append([x[-1] - y[-1] for x, y in zip(run["results"][metric_type_combo.get()]["training"],
                                                             run["results"][metric_type_combo.get()]["testing"])])
                names.append(run["method"])
    elif set_type_combo.get() == "differences in training and validation":
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                values.append([x[-1] - y[-1] for x, y in zip(run["results"][metric_type_combo.get()]["training"],
                                                             run["results"][metric_type_combo.get()]["validation"])])
                names.append(run["method"])
    else:
        for run in dataset['runs']:
            if find_and_checked_checkbox(run['method']):
                values.append([])
                for fold in run["results"][metric_type_combo.get()][set_type_combo.get()]:
                    values[-1].append(fold[-1])
                names.append(run["method"])

    print("Dataset " + dataset_combo.get())
    for count_one, name_one in enumerate(names):
        print("\t" + name_one)
        for count_two, name_two in enumerate(names):
            if not (combos.__contains__(name_one + name_two)):
                u_statistic, p_value = stats.mannwhitneyu(values[count_one], values[count_two], alternative='two-sided')
                print(
                    f"\t\t{name_two}: U-statistic = {u_statistic}, p-value = {round(p_value, 3)}, reject = {p_value < 0.005}")
                combos.append(name_one + name_two)
                combos.append(name_two + name_one)


canvas = None
# Create the main window
window = tk.Tk()
window.title("Line Chart UI")

# Button to load a dataset
load_button = ttk.Button(window, text="Load Dataset", command=load_files)
load_button.grid(row=0, column=0, pady=10)

# Combo box for selecting columns
dataset_combo = ttk.Combobox(window, state='disabled')
dataset_combo.bind("<<ComboboxSelected>>", load_dataset)
dataset_combo.grid(row=0, column=1, pady=10)

# Combo box for selecting columns
chart_type_combo = ttk.Combobox(window, state='disabled')
chart_type_combo['values'] = ["Box and whisker", "Line chart"]
chart_type_combo.set("Box and whisker")
chart_type_combo.grid(row=0, column=2, pady=10)

# Combo box for selecting columns
metric_type_combo = ttk.Combobox(window, state='disabled')
metric_type_combo.grid(row=0, column=3, pady=10)

# Combo box for selecting columns
set_type_combo = ttk.Combobox(window, state='disabled', width=40)
set_type_combo.grid(row=0, column=4, pady=10)

# Button to plot the line chart
plot_button = ttk.Button(window, text="Plot Chart", command=plot_charts, state='disabled')
plot_button.grid(row=0, column=5, pady=10)

show_avg_and_std_button = ttk.Button(window, text="Show data table", command=create_table, state='disabled')
show_avg_and_std_button.grid(row=0, column=6, pady=10)

show_test_button = ttk.Button(window, text="Show Mann-Whitney U Test", command=create_test, state='disabled')
show_test_button.grid(row=0, column=7, pady=10)

save_button = ttk.Button(window, text="Save Chart", command=save_charts, state='disabled')
save_button.grid(row=0, column=8, pady=10)

# Button to exit the application
exit_button = ttk.Button(window, text="Exit", command=window.destroy)
exit_button.grid(row=0, column=9, pady=10)

regularisation_method_checkboxes = []
regularisation_method_checkboxes_values = []
datasets = []

# Run the main loop
window.mainloop()
