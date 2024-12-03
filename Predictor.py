import numpy
import numpy as np
import toml
from matplotlib import pyplot as plt
from scipy.interpolate import interpolate, splev, interp1d

from utils import calculate_dist, ns_to_s, check_monotonity, get_first_not_monoton_el

with open('configuration.toml', 'r') as f:
    config = toml.load(f)


class Fifo:
    def __init__(self, max_size):
        if max_size <= 0:
            raise ValueError("maxSize must be greater than 0")
        self.maxSize = max_size
        self.data = []
        self.last = None
        self.last_index = None

    def add(self, item):
        if len(self.data) > self.maxSize:
            raise ValueError("Somehow u managed to add more items to the list, than its possible")
        if len(self.data) == self.maxSize:
            self.data.pop(0)
            self.data.append(item)
            self.last = item
            self.last_index = len(self.data) - 1
            return False
        self.data.append(item)
        self.last = item
        self.last_index = len(self.data) - 1
        return True

    def pop(self):
        if len(self.data) > 0:
            self.data.pop(0)
            return True
        else:
            raise IndexError("There's no more items in the list :(")

    def get(self, i):
        if i == self.maxSize - 1:
            return self.last
        return self.data[i]

    def get_data_part(self, i):
        result = []
        for j in self.data:
            result.append(j[i])
        return result

    def getLast(self):
        return self.last

    def getLastIndex(self):
        return self.last_index

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def get_prediction(x, y, modified_times):
    if not config['predictor']['use_strict_method']:
        x_poly = np.poly1d(np.polyfit(modified_times, x, 3))  # Másodfokú polinomiális illesztés
        y_poly = np.poly1d(np.polyfit(modified_times, y, 3))

        next_time = modified_times[len(modified_times) - 1] + 3
        x_pred = int(x_poly(next_time))
        y_pred = int(y_poly(next_time))
        print("pred: ", x_pred, y_pred)
        return x_pred, y_pred
    else:
        if check_monotonity(x) and check_monotonity(y):
            x_poly = np.poly1d(np.polyfit(modified_times, x, 3))
            y_poly = np.poly1d(np.polyfit(modified_times, y, 3))

            next_time = modified_times[len(modified_times) - 1] + 3
            x_pred = int(x_poly(next_time))
            y_pred = int(y_poly(next_time))
            print("pred: ", x_pred, y_pred)
            return x_pred, y_pred
        elif not check_monotonity(x):
            print("x is not monotonically increasing")
            i = get_first_not_monoton_el(x)
            if i < round(len(x) / 2):
                x = x[i:]
                y = y[i:]
                modified_times = modified_times[i:]
                x_poly = np.poly1d(np.polyfit(modified_times, x, 3))  # Másodfokú polinomiális illesztés
                y_poly = np.poly1d(np.polyfit(modified_times, y, 3))

                next_time = modified_times[len(modified_times) - 1] + 3
                x_pred = int(x_poly(next_time))
                y_pred = int(y_poly(next_time))
                print("pred: ", x_pred, y_pred)
                return x_pred, y_pred
            else:
                x = x[:i]
                y = y[:i]
                modified_times = modified_times[:i]
                x_poly = np.poly1d(np.polyfit(modified_times, x, 3))  # Másodfokú polinomiális illesztés
                y_poly = np.poly1d(np.polyfit(modified_times, y, 3))

                next_time = modified_times[len(modified_times) - 1] + 3
                x_pred = int(x_poly(next_time))
                y_pred = int(y_poly(next_time))
                print("pred: ", x_pred, y_pred)
                return x_pred, y_pred
        elif not check_monotonity(y):
            print("y is not monotonically increasing")
            j = get_first_not_monoton_el(y)
            if j < round(len(y) / 2):
                y = y[j:]
                x = x[j:]
                modified_times = modified_times[j:]
                x_poly = np.poly1d(np.polyfit(modified_times, x, 3))  # Másodfokú polinomiális illesztés
                y_poly = np.poly1d(np.polyfit(modified_times, y, 3))

                next_time = modified_times[len(modified_times) - 1] + 3
                x_pred = int(x_poly(next_time))
                y_pred = int(y_poly(next_time))
                print("pred: ", x_pred, y_pred)
                return x_pred, y_pred
            else:
                y = y[:j]
                x = x[:j]
                modified_times = modified_times[:j]
                x_poly = np.poly1d(np.polyfit(modified_times, x, 3))  # Másodfokú polinomiális illesztés
                y_poly = np.poly1d(np.polyfit(modified_times, y, 3))

                next_time = modified_times[len(modified_times) - 1] + 3
                x_pred = int(x_poly(next_time))
                y_pred = int(y_poly(next_time))
                print("pred: ", x_pred, y_pred)
                return x_pred, y_pred
        else:
            raise ValueError("I have no idea whats going on")


class Predictor:
    def __init__(self, max_size):
        self.fifo = Fifo(max_size)
        self.has_prediction = False
        self.speed = None

    def add(self, item):
        self.fifo.add(item)
        if len(self.fifo.data) == self.fifo.maxSize:
            self.has_prediction = True
        else:
            self.has_prediction = False

    def getTimes(self):
        print(len(self.fifo.data))
        times = []
        for i in self.fifo.data:
            print("---")
            times.append(i[1])

        return times

    def predict(self):
        if not self.has_prediction:
            raise ValueError("Not enough points to make a prediction.")
        print("speed: ", self.get_speed())
        points = np.array(self.fifo.get_data_part(0))
        times = self.getTimes()
        base_value = int(str(times[0])[:7])

        modified_times = [round(t - base_value * 10**(len(str(times[0]).split('.')[0]) - 7), 1) for t in times]

        if len(times) != len(points):
            raise ValueError("Mismatch between times and points lengths.")
        if not np.all(np.diff(times) > 0):
            # raise ValueError("Times are not in increasing order.")
            print("Times are not in increasing order.")

        x = points[:, 0]
        y = points[:, 1]

        print("x", x)
        print("y", y)

        return get_prediction(x, y, modified_times)

    def get_speed(self):
        prev_item = self.fifo.get(self.fifo.getLastIndex() - 1)
        current_item = self.fifo.getLast()
        distance = calculate_dist(prev_item[0], current_item[0])
        delta_time_s = (current_item[1] - prev_item[1]) / 1e9
        if delta_time_s == 0:
            return 0

        return distance / delta_time_s







