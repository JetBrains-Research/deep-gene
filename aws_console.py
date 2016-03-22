#!/usr/bin/python2

import os
import subprocess


server = ""
key = ""


def show_help():
    print """
s [server]      - set server
k [path to pem] - set key file
m               - move all data to server
c               - connect to  server
q               - quit console
"""


def read_properties():
    global server, key
    if os.path.exists('.aws-config'):
        f = open('.aws-config', 'r')
        lines = f.readlines()
        server = lines[0][:-1]
        key = lines[1][:-1]
        f.close()


def write_properties():
    f = open('.aws-config', 'w')
    f.write(server + "\n")
    f.write(key + "\n")
    f.close()


def main():
    global server, key
    print("AWS console")

    read_properties()

    print("Server:[{}]".format(server))
    print("Key:[{}]".format(key))

    while True:
        user_input = raw_input('>')

        if user_input.startswith("s"):
            server = user_input.split(" ")[1]
            write_properties()

        if user_input.startswith("c"):
            subprocess.call(["ssh", "-i", key, "-o", "UserKnownHostsFile=/dev/null", "-o", "StrictHostKeyChecking=no",
                             "ubuntu@" + server])

        if user_input.startswith("m"):
            subprocess.call(["scp", "-i", key,  "-o", "UserKnownHostsFile=/dev/null", "-o", "StrictHostKeyChecking=no",
                             "-r", "../deep-gene/", "ubuntu@" + server + ":~"])

        if user_input.startswith("k"):
            key = user_input.split(" ")[1]
            write_properties()

        if user_input == "q":
            print("Exit")
            break

        if user_input == "h":
            show_help()


if __name__ == '__main__':
    main()