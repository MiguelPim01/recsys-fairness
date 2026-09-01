"""Terminal-aware styling for pipeline output."""

import logging
import os
import sys
from enum import Enum
from typing import TextIO

from tqdm.auto import tqdm


class ConsoleColor(str, Enum):
    YELLOW = "\033[93m"


BOLD = "\033[1m"
RESET = "\033[0m"


def supports_color(stream: TextIO) -> bool:
    """Return whether ANSI styles should be emitted for a stream."""
    return "NO_COLOR" not in os.environ and bool(
        getattr(stream, "isatty", lambda: False)()
    )


def style_text(
    text: str,
    color: ConsoleColor | None = None,
    *,
    bold: bool = False,
    stream: TextIO | None = None,
) -> str:
    """Style text when its destination supports terminal colors."""
    output_stream = stream or sys.stdout
    if not supports_color(output_stream) or (color is None and not bold):
        return text

    prefix = f"{BOLD if bold else ''}{color.value if color else ''}"
    return f"{prefix}{text}{RESET}"


def styled_print(
    text: str,
    color: ConsoleColor | None = None,
    *,
    bold: bool = False,
    file: TextIO | None = None,
) -> None:
    """Print text with terminal-aware styling."""
    output_stream = file or sys.stdout
    print(
        style_text(text, color, bold=bold, stream=output_stream),
        file=output_stream,
    )


def styled_tqdm(iterable, color: ConsoleColor | None = None, **kwargs):
    """Create a tqdm progress bar whose complete line uses a stage color."""
    output_stream = kwargs.get("file") or sys.stderr
    if color is not None and supports_color(output_stream):
        bar_format = kwargs.get("bar_format") or "{l_bar}{bar}{r_bar}"
        kwargs["bar_format"] = f"{color.value}{bar_format}{RESET}"

    return tqdm(iterable, **kwargs)


class StyledFormatter(logging.Formatter):
    """Apply terminal-aware styling to complete log messages."""

    def __init__(
        self,
        color: ConsoleColor,
        stream: TextIO,
        *,
        bold: bool = False,
        fmt: str | None = None,
    ) -> None:
        super().__init__(fmt)
        self.color = color
        self.stream = stream
        self.bold = bold

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return style_text(
            message,
            self.color,
            bold=self.bold,
            stream=self.stream,
        )
