"""
Compatibility shim for the removed stdlib `imghdr` module (Python 3.13+).

Streamlit imports `imghdr` to detect basic image types. This lightweight
implementation uses Pillow so older Streamlit versions keep working.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Union

from PIL import Image


def what(file: Union[str, bytes, "os.PathLike[str]"], h: Optional[bytes] = None) -> Optional[str]:
    """
    Roughly matches the old imghdr.what interface.

    - `file`: path or file-like object (Streamlit passes a path)
    - `h`: optional bytes; if provided we detect from in-memory data
    """
    try:
        if h is not None:
            with Image.open(BytesIO(h)) as img:
                return img.format.lower()

        with Image.open(file) as img:
            return img.format.lower()
    except Exception:
        return None


__all__ = ["what"]

