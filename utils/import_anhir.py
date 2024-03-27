import requests
from requests.auth import HTTPBasicAuth
import shutil
import pandas as pd
import os

def createNewPath(newPath):
    if not os.path.exists(newPath):
        os.mkdir(newPath)

def getSave(url, out):
    r=requests.get(url, stream=True, auth=HTTPBasicAuth(username, password))
    with open(out, 'wb') as handler:
        handler.write(r.content)
    del r

username = 'ANHIR-guest'
password = 'isbi2019'
sheet_url = "D:/codes/media/data/dataset_medium.csv"
baseurl = "http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/landmarks"
outbase = "F:/N002-Research/liver-pathology/anhir/medium"

urls = pd.read_csv(sheet_url) 

subfolders = [ 'images', 'landmarks' ]

for subfolder in subfolders:
    createNewPath(f'{outbase}/{subfolder}')

for i, row in urls.iterrows():

    source_image = row["Source image"]
    target_image = row["Target image"]

    source_landmarks = row["Source landmarks"]
    target_landmarks = row["Target landmarks"]

    """     images = [source_image, target_image]

        for j in range(len(images)): 
            
            filesuburl = images[j]
            filename = filesuburl.replace('/', '_')
            url = f"{baseurl}/{filesuburl}"
            out = f'{outbase}/images/{filename}'
            if not os.path.exists(out):
                getSave(url, out)
    """

    landmarks = [ source_landmarks, target_landmarks ]

    for j in range(len(landmarks)): 
        
        filesuburl = landmarks[j]
        filename = filesuburl.replace('/', '_')
        url = f"{baseurl}/{filesuburl}"
        print(url)

        out = f'{outbase}/landmarks/{filename}'
        if not os.path.exists(out):
            getSave(url, out)