#!/usr/bin/env python3
"""
X-Ray Fracture Detection GUI Launcher
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from gui_app import main
    main()
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements_gui.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error running application: {e}")
    sys.exit(1)