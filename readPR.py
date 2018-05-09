import os
from os import path, listdir
import html2text 
from subprocess import call
import fnmatch

pressrelease_folders = ["step2/"]
pressrelease_folders_txt = ["step2_txt/"]
prefix = path.expanduser("~/research/nlp/data/")


i = -1
for folder in pressrelease_folders:
    i += 1
    print("Copying from ", folder ," to ", pressrelease_folders_txt[i])
    fullpath = path.join(prefix, folder)
    fullpath_txt = path.join(prefix, pressrelease_folders_txt[i])
    totFilesInFolder = len(fnmatch.filter(os.listdir(fullpath),
    '*.txt'))
    countFiles = 0
    for f in listdir(path.join(prefix, folder)):
        countFiles += 1
        newname = fullpath_txt + f 
        if os.path.isfile(newname):
            continue
        fullname = fullpath + f
        if countFiles % 1000 == 0:
            print("[{0:8d}/{1:8d} ] = {2:s}".format(countFiles, totFilesInFolder,
            fullcommand))
        try:
            text = open(fullname).readlines()

            fullcommand = "inscript.py " + fullname + " -o " + newname
            #  print("[{0:8d}/{1:8d} ] = {2:s}".format(countFiles, totFilesInFolder,
            #  fullcommand))
            os.system(fullcommand)
        except ValueError:
            print("*** ERROR WITH ", fullname)
            continue

        #  if countFiles > 999:
        #      break


