import sys
import cv2
import os

import numpy as np
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QApplication, QHBoxLayout, QSlider
from PyQt6.QtWidgets.QWidget import height


class VideoThread(QThread):
    resultSignal = pyqtSignal(object)

    def __init__(self, videoPath):
        super().__init__()
        self.videoPath = videoPath
        self.running = True
        self.pauseThread = False

    def run(self):
        video = cv2.VideoCapture(self.videoPath)

        if not video.isOpened():
            return

        subtractorBackground =cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=80,
            detectShadows=True)

        kernelOpen = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernelClose = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))

        while self.running:

            if self.pauseThread:
                self.msleep(50)
                continue

            ret, frame = video.read()

            if not ret:
                break

            height, width = frame.shape[:2]
            maskRoi = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(maskRoi, (0, int(height * 0.4)), (width, height), 255, -1)
            roiFrame = cv2.bitwise_and(frame, frame, mask=maskRoi)

            grayFrame = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2GRAY)
            blurFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)

            fgMask = subtractorBackground.apply(blurFrame, learningRate=0.01)
            _, fgMask = cv2.threshold(fgMask, 240, 255, cv2.THRESH_BINARY)

            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernelOpen, iterations=1)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernelClose, iterations=3)

            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)

                    ratio = w/h
                    if 0.4 <  ratio < 4.0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ret:
                self.resultSignal.emit(frame)

            self.msleep(30)


        video.release()

    def pause(self):
        self.pauseThread = not self.pauseThread


    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.thread = None
        self.videoPath = None

        self.mainLayout = QHBoxLayout()

        self.videoWidget = QVideoWidget()

        self.videoPlayer = QMediaPlayer()
        self.videoPlayer.setVideoOutput(self.videoWidget)

        self.videoPlayer.durationChanged.connect(self.updateDuration)
        self.videoPlayer.positionChanged.connect(self.updatePosition)

        self.slider = QSlider()
        self.slider.sliderMoved.connect(self.setPosition)

        # self.videoPlayer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoWidget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.videoWidget.setMinimumSize(800, 600)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,0)


        self.rightLayout = QVBoxLayout()

        self.btnLoadVideo = QPushButton("Load Video")
        self.btnLoadVideo.clicked.connect(self.loadVideo)

        self.btnPlayResume = QPushButton("Play/Resume")
        self.btnPlayResume.clicked.connect(self.playResume)

        self.btnPause = QPushButton("Pause")
        self.btnPause.clicked.connect(self.pause)

        self.btnStop = QPushButton("Stop")
        self.btnStop.clicked.connect(self.stop)

        self.rightLayout.addStretch()

        self.rightLayout.addWidget(self.btnLoadVideo)
        self.rightLayout.addWidget(self.btnPlayResume)
        self.rightLayout.addWidget(self.btnPause)
        self.rightLayout.addWidget(self.btnStop)

        self.rightLayout.addStretch()

        self.mainLayout.addWidget(self.videoWidget)
        self.mainLayout.addWidget(self.slider)
        self.mainLayout.addLayout(self.rightLayout)

        self.setLayout(self.mainLayout)

    def updateDuration(self, duration):
        self.slider.setRange(0, duration)

    def updatePosition(self, position):
        self.slider.setValue(position)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def loadVideo(self):
        videoPath, _ = QFileDialog.getOpenFileName()

        if videoPath:
            self.videoPath = videoPath
            video = cv2.imread(videoPath)
            self.showVideo(video)

    def playResume(self):
        print("Play/Resume button clicked!(Video logic not implemented yet)")

    def pause(self):
        print("Pause button clicked!(Video logic not implemented yet)")

    def stop(self):
        print("Stop button clicked!(Video logic not implemented yet)")









app = QApplication(sys.argv)

window = App()
window.show()

app.exec()