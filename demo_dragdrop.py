#!/usr/bin/env python3
"""
Enhanced X-Ray GUI with Drag & Drop Support
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

if __name__ == "__main__":
    try:
        from gui_app import main
        print("🚀 Launching X-Ray Analysis GUI with Drag & Drop...")
        print("✨ New Features:")
        print("   📂 Drag & drop X-ray images directly onto the input panel")
        print("   🖱️ Visual feedback when dragging files")
        print("   📁 Supports JPG, PNG, JPEG, BMP, TIFF formats")
        print("   ✅ Works alongside the existing upload button")
        print("")
        print("💡 Usage:")
        print("   • Drag any X-ray image file onto the left input panel")
        print("   • Or use the 'Upload Image' button as before")
        print("   • The panel will highlight when you drag a valid image over it")
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