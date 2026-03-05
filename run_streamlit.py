"""
Launcher for Streamlit so that the project root is on sys.path first.
This lets our imghdr compatibility shim be found when Streamlit imports imghdr
(Python 3.13+ removed the stdlib imghdr module).
"""
import sys
import os

# Project root must be first so "import imghdr" finds our imghdr.py
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Now run Streamlit (pass through any args, e.g. "run", "app.py")
from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    main()
