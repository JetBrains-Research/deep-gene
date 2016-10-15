import os
import time


def human_time(t):
    seconds = int(t)
    milliseconds = t - int(t)

    hours = seconds // (60 * 60)
    minutes = (seconds // 60) % 60
    seconds %= 60

    result = ""
    if hours > 0:
        result += str(hours) + "h"
    if minutes > 0:
        result += str(minutes) + "m"

    if minutes >= 10 or hours > 0:
        result += str(seconds) + "s"
    else:
        result += "{:2.2f}s".format(seconds + milliseconds)
    return result


class FileLogger(object):
    def __init__(self, path, name):
        self.path = os.path.join(path, name + ".log")
        self.log_file = open(self.path, "w")
        self.start_time = time.time()

    def log(self, data):
        print(data)
        self.log_file.write(str(data) + "\n")
        self.log_file.flush()

    def close(self):
        end_time = time.time()
        self.log("Done in {}".format(human_time(end_time - self.start_time)))
        self.log_file.close()
        self.log_file = None

    def __del__(self):
        if self.log_file:
            print("Error: logger {} not closed.".format(self.path))


class PrintLogger(object):
    def __init__(self):
        pass

    def log(self, data):
        print(data)

    def close(self):
        pass

def get_result_directory_path(suffix):
    result_path = os.path.join("results", "{}_{}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), suffix))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path