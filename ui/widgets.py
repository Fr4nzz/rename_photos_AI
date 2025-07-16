# ai-photo-processor/ui/widgets.py

from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QPainter

class ClickableLabel(QLabel):
    """
    A custom QLabel that is clickable and preserves the aspect ratio of its pixmap
    by handling its own paint event. This is the definitive, robust implementation.
    """
    clicked = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file_path = ""
        self._pixmap = QPixmap()
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)

    def setFilePath(self, path: str):
        """Sets the file path that will be emitted when clicked."""
        self._file_path = path

    def mouseReleaseEvent(self, event):
        """Overrides the default mouse release event handler to emit the clicked signal."""
        if self._file_path:
            self.clicked.emit(self._file_path)

    def setPixmap(self, pixmap: QPixmap):
        """Stores the original pixmap and triggers a repaint."""
        self._pixmap = pixmap if pixmap else QPixmap()
        self.update() # Trigger a repaint, which will call paintEvent

    def hasHeightForWidth(self) -> bool:
        """Tells the layout that height depends on width."""
        return not self._pixmap.isNull()

    def heightForWidth(self, width: int) -> int:
        """Calculates the ideal height for a given width to maintain aspect ratio."""
        if self._pixmap.isNull() or self._pixmap.width() == 0:
            return self.height()
        return int(width * (self._pixmap.height() / self._pixmap.width()))

    def paintEvent(self, event):
        """
        Paints the pixmap scaled to the current widget size, preserving aspect ratio
        and centering the result.
        """
        super().paintEvent(event) # Draw background and border first
        
        if self._pixmap.isNull():
            return

        size = self.size()
        scaled_pixmap = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        x = (size.width() - scaled_pixmap.width()) / 2
        y = (size.height() - scaled_pixmap.height()) / 2

        painter = QPainter(self)
        painter.drawPixmap(int(x), int(y), scaled_pixmap)