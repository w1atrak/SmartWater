import csv, serial
import json
import time
from time import sleep
from neural_networking import predict, Sigmoid
arduino = serial.Serial("COM3", 9600)

previous_data = None
deltas = [0.1, 0.3, 0.1, 1, 0.1]  # time_passed, temp, humidity, light, soil_humidity


def read_from_arduino():
    line = arduino.readline().decode().strip()
    return json.loads(line)

def send_signal_arduino():
    arduino.write(b'W') # or sth else

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

    if previous_data is None or any(
        abs(data_line[i] - previous_data[i]) > threshhold
        for i, threshhold in enumerate(deltas)
    ):
        print(data_line)

        with open("../train_data.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_line)
        with open("../new.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_line)
        previous_data = data_line

        # predict function call
        prediction = predict(
            x=data_line[:-1],
            layers=[4,3],
            activation_function=Sigmoid()
        )
        if prediction[0] >= 0.9:
            send_signal_arduino()
