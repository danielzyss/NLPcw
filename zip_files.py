from zipfile import ZipFile
import os

for name in os.listdir("exports"):
    if name[-3:]=="txt":
        with ZipFile("zip-exports/"+name[:-4]+".zip", "w") as newzip:
            newzip.write("exports/"+name, arcname="predictions.txt")
