#!/usr/bin/env python3

import json
import random
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPoint, QParallelAnimationGroup, QPropertyAnimation, QRect, Qt, QTimer, QUrl, pyqtProperty
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
    QRadialGradient,
    QImage,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import io
import math
import struct
import tempfile
import os

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None
    _NUMPY = False
