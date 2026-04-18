import sys
import cv2
import time

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QApplication, QHBoxLayout, QSlider



class VideoThread(QThread):
    resultSignal = pyqtSignal(object)
    durationSignal = pyqtSignal(int)
    positionSignal = pyqtSignal(int)
    fpsSignal = pyqtSignal(int)

    def __init__(self, videoPath):
        super().__init__()
        self.videoPath = videoPath
        self.running = True
        self.pauseThread = False

    def run(self):
        video = cv2.VideoCapture(self.videoPath)

        if not video.isOpened():
            return

        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.durationSignal.emit(totalFrames)

        subtractorBackground =cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=80,
            detectShadows=True)

        kernelOpen = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernelClose = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))

        prevTime = time.time()
        self.newPosition = -1

        while self.running:
            if self.newPosition != -1:
                video.set(cv2.CAP_PROP_POS_FRAMES, self.newPosition)
                self.newPosition = -1

            if self.pauseThread:
                self.msleep(50)

                prevTime = time.time()
                continue

            ret, frame = video.read()

            if not ret:
                break

            currentTime = time.time()
            elapsedTime = currentTime - prevTime
            if elapsedTime > 0:
                fps = int(1 / elapsedTime)
                self.fpsSignal.emit(fps)
            prevTime = currentTime

            currentFrame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            self.positionSignal.emit(currentFrame)

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frameForDisplay = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)

            height, width = frame.shape[:2]
            maskRoi = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(maskRoi, (0, int(height * 0.4)), (width, height), 255, -1)

            roiFrame = cv2.bitwise_and(grayFrame, grayFrame, mask=maskRoi)
            blurFrame = cv2.GaussianBlur(roiFrame, (5, 5), 0)

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
                        cv2.rectangle(frameForDisplay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frameForDisplay, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ret:
                self.resultSignal.emit(frameForDisplay)

            self.msleep(30)


        video.release()

    def pause(self):
        self.pauseThread = not self.pauseThread


    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def setFrame(self, frameNumber):
        self.newPosition = frameNumber

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.thread = None
        self.videoPath = None

        self.mainLayout = QHBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setMinimumSize(800, 600)

        self.fpsLabel = QLabel("FPS: 0")
        self.fpsLabel.setStyleSheet("color: #00FF00; font-size: 18px; font-weight: bold;")
        self.fpsLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderReleased.connect(self.seekVideo)
        self.slider.setRange(0,0)

        self.btnLoadVideo = QPushButton("Load Video")
        self.btnLoadVideo.clicked.connect(self.loadVideo)

        self.btnPlayPause = QPushButton("Play/Pause")
        self.btnPlayPause.clicked.connect(self.playPause)

        self.btnStop = QPushButton("Stop")
        self.btnStop.clicked.connect(self.stop)

        self.leftLayout.addWidget(self.fpsLabel)
        self.leftLayout.addWidget(self.videoLabel)
        self.leftLayout.addWidget(self.slider)

        self.rightLayout.addStretch()

        self.rightLayout.addWidget(self.btnLoadVideo)
        self.rightLayout.addWidget(self.btnPlayPause)
        self.rightLayout.addWidget(self.btnStop)

        self.rightLayout.addStretch()

        self.mainLayout.addLayout(self.leftLayout)
        self.mainLayout.addLayout(self.rightLayout)

        self.setLayout(self.mainLayout)

    def updateDuration(self, duration):
        self.slider.setRange(0, duration)

    def updatePosition(self, position):
        if not self.slider.isSliderDown():
            self.slider.setValue(position)

    def loadVideo(self):
        videoPath, _ = QFileDialog.getOpenFileName()

        if videoPath:
            self.videoPath = videoPath
            self.videoLabel.setText("Video loaded. Press Play/Pause.")

            if self.thread is not None:
                self.thread.stop()

    def playPause(self):
        if not getattr(self, 'videoPath', None):
            print("First, upload the video!")
            return

        if self.thread is None or not self.thread.isRunning():

            self.thread = VideoThread(self.videoPath)

            self.thread.resultSignal.connect(self.showVideo)
            self.thread.durationSignal.connect(self.updateDuration)
            self.thread.positionSignal.connect(self.updatePosition)
            self.thread.fpsSignal.connect(self.updateFps)

            self.thread.start()
            self.videoLabel.clear()

        else:
            self.thread.pause()

    def stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()

            self.slider.setValue(0)
            self.fpsLabel.setText("FPS:0")

            self.videoLabel.clear()



    def showVideo(self, frame):
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w

        qtImage = QImage(
            rgbImage.data,
            w,
            h,
            bytesPerLine,
            QImage.Format.Format_RGB888
        ).copy()

        self.videoLabel.setPixmap(QPixmap.fromImage(qtImage).scaled(
            self.videoLabel.width(),
            self.videoLabel.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def updateFps(self, fps):
        self.fpsLabel.setText(f"FPS: {fps}")

    def seekVideo(self):
        if self.thread and self.thread.isRunning():
            position = self.slider.value()
            self.thread.setFrame(position)

app = QApplication(sys.argv)

window = App()
window.show()

app.exec()