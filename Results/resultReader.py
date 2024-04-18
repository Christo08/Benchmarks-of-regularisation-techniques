import math
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import json


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
                datasetNames.append(dataset["datasetName"])
                dataset_combo['values'] = datasetNames
                dataset_combo['state'] = 'normal'  # Set default value
        except pd.errors.EmptyDataError:
            print("Selected file is empty.")
        except pd.errors.ParserError:
            print("Error parsing the selected file. Please make sure it's a valid CSV file.")
    else:
        print("No file selected.")


def findAndCheckCheckbox(name):
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
            if object['datasetName'] == dataset_combo.get():
                dataset = object
                break

        value_type = ["accuracies", "losses"]
        value_type_combo['values'] = value_type
        value_type_combo.set(value_type[0])
        value_type_combo['state'] = 'normal'

        value = ["Difference in Testing and Training", "Difference in Validation and Training", "testing", "training",
                 "validation"]
        value_combo['values'] = value
        value_combo.set(value[0])
        value_combo['state'] = 'normal'
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
    except pd.errors.EmptyDataError:
        print("Selected file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the selected file. Please make sure it's a valid CSV file.")


def plot_line_chart():
    global dataset, lastRow
    selected_column = dataset_combo.get()
    yValue = []
    labels = []

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))
    if value_combo.get() == "Difference in Testing and Training":
        for run in dataset['runs']:
            if findAndCheckCheckbox(run['method']):
                first = True
                for fold in zip(run["results"][value_type_combo.get()]["training"],
                                run["results"][value_type_combo.get()]["testing"]):
                    if first:
                        yValue.append([y - x for x, y in zip(fold[0], fold[1])])
                        first = False
                    else:
                        yValue[-1] = [x + y for x, y in zip(yValue[-1], [a - b for a, b in zip(fold[0], fold[1])])]
                yValue[-1] = [x / len(run["results"][value_type_combo.get()]["training"]) for x in yValue[-1]]
                labels.append(run['method'])
    elif value_combo.get() == "Difference in Validation and Training":
        for run in dataset['runs']:
            if findAndCheckCheckbox(run['method']):
                first = True
                for fold in zip(run["results"][value_type_combo.get()]["training"],
                                run["results"][value_type_combo.get()]["validation"]):
                    if first:
                        yValue.append([y - x for x, y in zip(fold[0], fold[1])])
                        first = False
                    else:
                        yValue[-1] = [x + y for x, y in zip(yValue[-1], [a - b for a, b in zip(fold[0], fold[1])])]
                yValue[-1] = [x / len(run["results"][value_type_combo.get()]["training"]) for x in yValue[-1]]
                labels.append(run['method'])
    else:
        for run in dataset['runs']:
            if findAndCheckCheckbox(run['method']):
                first = True
                for fold in run["results"][value_type_combo.get()][value_combo.get()]:
                    if first:
                        yValue.append(fold)
                        first = False
                    else:
                        yValue[-1] = [x + y for x, y in zip(yValue[-1], fold)]
                yValue[-1] = [x / len(run["results"][value_type_combo.get()][value_combo.get()]) for x in yValue[-1]]
                labels.append(run['method'])

    epochs = [i + 1 for i in range(len(yValue[0]))]
    for index in range(len(yValue)):
        ax.plot(epochs, yValue[index], label=labels[index])

    # Set labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel(value_combo.get() + " " + value_type_combo.get())
    ax.set_title(selected_column + ": Epoch vs " + value_combo.get() + " " + value_type_combo.get())

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    # Display the plot on the canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=lastRow + 1, column=0, columnspan=6)


def plot_box_chart():
    global dataset, lastRow
    xValue = []
    yValue = []

    if value_combo.get() == "Difference in Testing and Training":
        for run in dataset['runs']:
            if findAndCheckCheckbox(run['method']):
                xValue.append([y[-1] - x[-1] for x, y in zip(run["results"][value_type_combo.get()]["training"],
                                                             run["results"][value_type_combo.get()]["testing"])])
                yValue.append(run['method'])
    elif value_combo.get() == "Difference in Validation and Training":
        for run in dataset['runs']:
            if findAndCheckCheckbox(run['method']):
                xValue.append([y[-1] - x[-1] for x, y in zip(run["results"][value_type_combo.get()]["training"],
                                                             run["results"][value_type_combo.get()]["validation"])])
                yValue.append(run['method'])
    else:
        for run in dataset['runs']:
            if findAndCheckCheckbox(run['method']):
                xValue.append([])
                for fold in run["results"][value_type_combo.get()][value_combo.get()]:
                    xValue[-1].append(fold[-1])
                yValue.append(run['method'])
    fig, ax = plt.subplots(figsize=(15, 8))

    bp = ax.boxplot(xValue, patch_artist=True,
                    notch='True', vert=0)
    ax.set_yticklabels(yValue)

    plt.title(
        "Box and Whisker chart of regularisation techniques for the " + value_combo.get() + " " + value_type_combo.get())
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=lastRow + 1, column=0, columnspan=6)


def plot_charts():
    if chart_type_combo.get() == "Box and Whisker":
        plot_box_chart()
    else:
        plot_line_chart()


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
chart_type_combo['values'] = ["Box and Whisker", "Line chart"]
chart_type_combo.set("Box and Whisker")
chart_type_combo.grid(row=0, column=2, pady=10)

# Combo box for selecting columns
value_type_combo = ttk.Combobox(window, state='disabled')
value_type_combo.grid(row=0, column=3, pady=10)

# Combo box for selecting columns
value_combo = ttk.Combobox(window, state='disabled', width=40)
value_combo.grid(row=0, column=4, pady=10)

# Button to plot the line chart
plot_button = ttk.Button(window, text="Plot Line Chart", command=plot_charts, state='disabled')
plot_button.grid(row=0, column=5, pady=10)

# Button to exit the application
exit_button = ttk.Button(window, text="Exit", command=window.destroy)
exit_button.grid(row=0, column=6, pady=10)

regularisation_method_checkboxes = []
regularisation_method_checkboxes_values = []
datasets = []

# Run the main loop
window.mainloop()
