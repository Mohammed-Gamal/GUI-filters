import cv2
import numpy as np
import sys
from os import path

from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog
from PyQt5.uic import loadUiType

from filters import Filters as f


# Import UI file
FORM_CLASS, QMainWindow = loadUiType(path.join(path.dirname(__file__), "filters.ui"))


class FiltersWindow(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(FiltersWindow, self).__init__(parent)
        self.setupUi(self)

        self.imPath = ''

        self.uploadBtn.clicked.connect(self.handleUploadBtn)

        self.comboBox_filter.currentIndexChanged.connect(self.start_operation)

    def handleUploadBtn(self):
        self.imPath = self.uploadFile(self.image_1)

    def start_operation(self, filter_index):
        if self.imPath == '':
            self.label_message.setText('Please upload an image first!')
            return

        # Read the image
        img = cv2.imread(self.imPath)

        # Apply the filter
        f.apply_filter(filter_index, img)

        # Display the message
        self.label_message.setText('Applying operation, please wait...')
        QTimer.singleShot(1500, lambda: self.label_message.setText(''))

        self.convert_cv_qt(img, self.image_2)

    def convert_cv_qt(self, img, label):
        if len(img.shape) == 2:  # Grayscale image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channel = img_rgb.shape
        q_image = QImage(img_rgb.data, width, height, width * channel, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)

        label.setPixmap(q_pixmap)


    def uploadFile(self, label: QLabel, dialog_title: str = "Select an image", filter_content: str = "JPG file (*.jpg)", filter_names=None):
        image_path = ''

        if filter_names is None:
            filter_names = ["All files (*.*)", "JPG file (*.jpg)", "PNG file (*.png)"]

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setWindowTitle(dialog_title)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.selectNameFilter(filter_content)
        file_dialog.setNameFilters(filter_names)

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            image_path = selected_files[0]

            # display the image
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.setStyleSheet("border: 5px double white;")

            return image_path
        else:
            print("No file selected!")
            return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FiltersWindow()
    window.show()
    app.exec()
