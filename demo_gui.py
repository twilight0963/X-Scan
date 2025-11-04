#!/usr/bin/env python3
"""
Enhanced X-Ray Fracture Detection GUI Demo
Modern, clean, and professionally styled interface
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

if __name__ == "__main__":
    try:
        from gui_app import main
        print("🚀 Launching Enhanced X-Ray Analysis GUI...")
        print("✨ Features:")
        print("   • Modern three-panel layout with perfect alignment")
        print("   • Rounded corners and subtle shadows")
        print("   • Responsive image scaling")
        print("   • Professional color scheme")
        print("   • Smooth hover effects on buttons")
        print("   • Clean typography and spacing")
        print("")
        main()
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("\n📦 Please install the required dependencies:")
        print("pip install -r requirements_gui.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)