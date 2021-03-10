# Optical Flow

![Demo Result](https://github.com/kranok-dev/Optical_Flow/blob/main/result_image.png?raw=true)

**Description**                                                               
> Simple application doing Lucas-Kanade's Optical Flow from "scratch" on a traffic video.

> This work is based on Khushboo Agarwal's code:
> https://github.com/khushboo-agarwal/Optical-Flow

**Installation**
> Clone this repository and the implemented code requires OpenCV, Numpy, Matplotlib and Scipy to be installed in Python (Python 3 was used):
  ```
  $ sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 python3-dev
  
  $ pip3 install opencv-contrib-python
  $ pip3 install numpy
  $ pip3 install matplotlib
  $ pip3 install scipy
  ```

**Execution**
> The application was designed to process an input video and saves the processed video in the same path:
```
$ python3 app.py

```

> Try the demo implemented, test other videos, and have fun!

**Demo Video**
> https://www.youtube.com/watch?v=BpSAWhDre3w
