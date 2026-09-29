from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from lautools.size_service import SizeService


class SizeServiceBridge(QObject):
    """Re-emits SizeService events as a Qt signal.

    The bridge lives in the GUI thread, so slots connected to `sizeEvent`
    run in the GUI thread even though events come from worker threads.
    """

    sizeEvent = Signal(object)

    def __init__(self, service: SizeService, parent=None):
        super().__init__(parent)
        self._unsubscribe = service.subscribe(self.sizeEvent.emit)

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
