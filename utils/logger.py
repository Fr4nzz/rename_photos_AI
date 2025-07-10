# ai-photo-processor/utils/logger.py

import datetime

class SimpleLogger:
    """A simple class to print formatted, timestamped messages to the console."""

    def _log(self, level: str, message: str):
        """Private base method for logging."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level.upper():<5}] {message}")

    def info(self, message: str):
        """For general progress and informational messages."""
        self._log("info", message)

    def warn(self, message: str):
        """For non-critical issues or potential problems."""
        self._log("warn", message)

    def error(self, message: str, exception: Exception = None):
        """For errors that occur, optionally including the exception details."""
        full_message = f"{message}"
        if exception:
            full_message += f" | Exception: {type(exception).__name__} - {exception}"
        self._log("error", full_message)