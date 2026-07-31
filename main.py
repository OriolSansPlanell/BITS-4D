#!/usr/bin/env python3
"""
main.py - BiTS 4D Application Entry Point

Launch the BiTS 4D application for bivariate tomography segmentation
of time-resolved 4D datasets.
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui import BiTS4DMainWindow


def main():
    """
    Main application entry point
    """
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent look
    
    # Create and show main window
    window = BiTS4DMainWindow()
    window.show()
    
    # Print startup info
    print("=" * 70)
    print("BiTS 4D - Bivariate Tomography Segmentation")
    print("Version: 2.0.0-beta")
    print("=" * 70)
    print("\nApplication started successfully!")
    print("\nFeatures:")
    print("  • Dual histogram display (Global + Local)")
    print("  • Time navigation with playback")
    print("  • ROI-based segmentation (polygon and rectangle)")
    print("  • GPU acceleration for histogram computation")
    print("  • Progress indicators for long operations")
    print("\nTo get started:")
    print("  1. File → Load 4D Dataset")
    print("  2. Draw ROI on global histogram")
    print("  3. Segment current or all timepoints")
    print("=" * 70)
    
    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
