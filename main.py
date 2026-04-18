import sys
import cv2
import os

from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QApplication, QHBoxLayout, QSlider


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