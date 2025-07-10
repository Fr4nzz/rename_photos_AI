# ai-photo-processor/ui/widgets.py

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap

class ClickableLabel(QLabel):
    """
    A custom QLabel that emits a 'clicked' signal containing a file path
    when the user clicks on it.
    """
    clicked = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file_path = ""
        self.setCursor(Qt.PointingHandCursor)
        self.setWordWrap(True)

    def setFilePath(self, path: str):
        """Sets the file path that this label will open when clicked."""
        self._file_path = path

    def mouseReleaseEvent(self, event):
        """Overrides the default mouse release event handler."""
        if self._file_path:
            self.clicked.emit(self._file_path)

    def setPixmap(self, pixmap: QPixmap):
        """Override setPixmap to scale to the label's size."""
        if pixmap and not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
        else:
            super().setPixmap(QPixmap())