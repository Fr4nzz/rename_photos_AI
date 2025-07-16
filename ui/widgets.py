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
        """Store pixmap and ask the layout to recalc geometry."""
        self._pixmap = pixmap if pixmap else QPixmap()
        self.updateGeometry()          # <- important for height-for-width
        self.update()                  # repaint

    # 1️⃣  Tell Qt that height follows width
    def hasHeightForWidth(self) -> bool:
        return not self._pixmap.isNull()

    def heightForWidth(self, width: int) -> int:
        """
        Return the height needed to keep the original aspect ratio when the
        layout gives us 'width' pixels.
        """
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
        # 2️⃣  Fit the pixmap fully inside the label while preserving aspect ratio
        scaled_pixmap = self._pixmap.scaled(
            size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        
        # Centre the pixmap in the label's rect
        x = (size.width()  - scaled_pixmap.width())  // 2
        y = (size.height() - scaled_pixmap.height()) // 2

        painter = QPainter(self)
        painter.drawPixmap(int(x), int(y), scaled_pixmap)