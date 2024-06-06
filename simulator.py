import tkinter as tk
from tkinter import messagebox

from activation_functions import Sigmoid
from neural_networking import predict


def run_predict():
    values = {label: entry.get() for label, entry in entries.items()}
    print(values)
    x = []
    # podajemy godziny
    x.append(int(values["time_passed"]) * 60 * 60 / 604_800)
    x.append((float(values["temp"]) - 10) / 30)
    # poniższe 3 podajemy procent, na przykład 27
    x.append(int(values["humidity"]) / 100)
    x.append(
        int(values["light"]) / 100
    )  # być może trzeba będzie dać (1 - to co jest teraz)
    x.append(int(values["soil_humidity"]) / 100)
    print(x)
    result = predict(x, [4, 4, 4], Sigmoid())
    messagebox.showinfo("Prediction Result", result)
    # btw komentarze pisał człowiek


root = tk.Tk()

labels = ["time_passed", "temp", "humidity", "light", "soil_humidity"]
legends = ["hours", "C", "%", "%", "%"]
entries = {}

for i, label_text in enumerate(labels):
    frame = tk.Frame(root)
    frame.pack()

    label = tk.Label(frame, text=f"{label_text} {legends[i]}: ")
    label.pack(side=tk.LEFT)

    entry = tk.Entry(frame)
    entry.pack(side=tk.LEFT)
    entries[label_text] = entry

button = tk.Button(root, text="Predict", command=run_predict)
button.pack()

root.mainloop()
