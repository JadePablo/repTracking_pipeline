# repTracking_pipeline

This repo contain python scripts that make up an etl pipeline that transfers my computer vision recordings into my own database.

## Overview:

I track my bicep curls using my webcam. Process the raw recordings to get insightful data about my rep performance. Then lob it into a database, where it creates dashboards about certain metrics and emails them to me on a weekly basis.

This entire process is automated.

## What I did and how I did it:

Bicep Curl Detection: cv2 (opencv) , mediapipe , numpy

Cleaning / Transforming Data: pandas , numpy

Loading Data: pymongo

Database: MongoDB

Real-Time Dashboards / Automated Emailing: MongoDB Charts

## Sample Dashboard

