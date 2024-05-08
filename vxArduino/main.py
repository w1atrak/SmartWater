import csv, serial
import json
import time
from time import sleep

arduino = serial.Serial("COM3", 9600)


def read_from_arduino():
    line = arduino.readline().decode().strip()
    return json.loads(line)


last_watered = int(time.time()) - 304_800
while True:
    sleep(1)

    data = read_from_arduino()
    data_line = []

    time_passed = (now := int(time.time())) - last_watered
    data_line.append(time_passed / 604_800)
    if button := data["is_button_pressed"]:
        last_watered = now
    data_line.append((data["temperature"] - 10) / 30)
    data_line.append(data["humidity"] / 100)
    data_line.append(min(data["light"], 1000) / 1000)
    data_line.append(max(0, min(data["soil_humidity"], 1000)) / 1000)
    data_line.append(button)

    print(data_line)

    with open("train_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data_line)

    with open("../new.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data_line)