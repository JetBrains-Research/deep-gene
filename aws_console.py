#!/usr/bin/python2

import os
import subprocess


server = ""
key = ""

def help():
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

    while(True):
        input = raw_input('>')

        if input.startswith("s"):
            server = input.split(" ")[1]
            write_properties()

        if input.startswith("c"):
             subprocess.call(["ssh", "-i", key,  "ubuntu@" + server])

        if input.startswith("k"):
            key = input.split(" ")[1]
            write_properties()

        if input == "q":
            print("Exit")
            break

        if input == "h":
            help()

if __name__ == '__main__':
    main()