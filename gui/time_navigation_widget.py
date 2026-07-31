"""
time_navigation_widget.py - Time Navigation Controls

Widget for navigating through temporal dimension of 4D datasets
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QSpinBox, QLabel, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer


class TimeNavigationWidget(QWidget):
    """
    Widget for navigating through temporal dimension
    """
    
    # Signals
    timepoint_changed = pyqtSignal(int)
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    
    def __init__(self, num_timepoints=100, parent=None):
        super().__init__(parent)
        self.num_timepoints = num_timepoints
        self.is_playing = False
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._next_frame)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Timepoint info label
        self.info_label = QLabel(f"Timepoint: 0 / {self.num_timepoints - 1}")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Slider with navigation buttons
        slider_layout = QHBoxLayout()
        
        self.first_btn = QPushButton("⏮")
        self.first_btn.setMaximumWidth(40)
        self.first_btn.setToolTip("Jump to first timepoint")
        self.first_btn.clicked.connect(self.jump_to_first)
        slider_layout.addWidget(self.first_btn)
        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setMaximumWidth(40)
        self.prev_btn.setToolTip("Previous timepoint")
        self.prev_btn.clicked.connect(self.previous_timepoint)
        slider_layout.addWidget(self.prev_btn)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_timepoints - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_change)
        slider_layout.addWidget(self.slider)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setMaximumWidth(40)
        self.next_btn.setToolTip("Next timepoint")
        self.next_btn.clicked.connect(self.next_timepoint)
        slider_layout.addWidget(self.next_btn)
        
        self.last_btn = QPushButton("⏭")
        self.last_btn.setMaximumWidth(40)
        self.last_btn.setToolTip("Jump to last timepoint")
        self.last_btn.clicked.connect(self.jump_to_last)
        slider_layout.addWidget(self.last_btn)
        
        layout.addLayout(slider_layout)
        
        # Spinbox for direct entry
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(QLabel("Go to timepoint:"))
        
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximumWidth(100)
        self.spinbox.setMaximum(self.num_timepoints - 1)
        self.spinbox.valueChanged.connect(self._on_spinbox_change)
        spinbox_layout.addWidget(self.spinbox)
        spinbox_layout.addStretch()
        
        layout.addLayout(spinbox_layout)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setToolTip("Play/Pause animation")
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_btn)
        
        playback_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(1)
        self.fps_spinbox.setMaximum(60)
        self.fps_spinbox.setValue(10)
        self.fps_spinbox.setMaximumWidth(60)
        self.fps_spinbox.setToolTip("Frames per second")
        playback_layout.addWidget(self.fps_spinbox)
        
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.setToolTip("Loop playback")
        playback_layout.addWidget(self.loop_checkbox)
        
        playback_layout.addStretch()
        
        layout.addLayout(playback_layout)
        
        self.setLayout(layout)
    
    def set_num_timepoints(self, num_timepoints: int):
        """Update the number of timepoints"""
        self.num_timepoints = num_timepoints
        self.slider.setMaximum(num_timepoints - 1)
        self.spinbox.setMaximum(num_timepoints - 1)
        self._update_info_label()
    
    def _on_slider_change(self, value):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self._update_info_label()
        self.timepoint_changed.emit(value)
    
    def _on_spinbox_change(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._update_info_label()
        self.timepoint_changed.emit(value)
    
    def _update_info_label(self):
        current = self.slider.value()
        self.info_label.setText(f"Timepoint: {current} / {self.num_timepoints - 1}")
    
    def next_timepoint(self):
        current = self.slider.value()
        if current < self.num_timepoints - 1:
            self.slider.setValue(current + 1)
    
    def previous_timepoint(self):
        current = self.slider.value()
        if current > 0:
            self.slider.setValue(current - 1)
    
    def jump_to_first(self):
        self.slider.setValue(0)
    
    def jump_to_last(self):
        self.slider.setValue(self.num_timepoints - 1)
    
    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        fps = self.fps_spinbox.value()
        interval = int(1000 / fps)
        self.playback_timer.start(interval)
        self.playback_started.emit()
    
    def stop_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.playback_timer.stop()
        self.playback_stopped.emit()
    
    def _next_frame(self):
        current = self.slider.value()
        if current < self.num_timepoints - 1:
            self.slider.setValue(current + 1)
        elif self.loop_checkbox.isChecked():
            self.slider.setValue(0)
        else:
            self.stop_playback()
    
    def get_current_timepoint(self) -> int:
        return self.slider.value()
