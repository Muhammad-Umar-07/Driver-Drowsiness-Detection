import pygame
import threading
import time
import numpy as np
from collections import deque

class AlarmSystem:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.alarm_active = False
        self.alarm_thread = None
        self.create_beep_sound()
   
    def create_beep_sound(self):
        sample_rate = 44100
        duration = 1000  # 1-second duration
        frequency = 880
        n_samples = int(round(duration * 0.001 * sample_rate))
        buf = np.zeros((n_samples, 2), dtype=np.int16)
        for s in range(n_samples):
            t = float(s) / sample_rate
            buf[s][0] = int(32767 * 0.5 * np.sin(2 * np.pi * frequency * t))
            buf[s][1] = int(32767 * 0.5 * np.sin(2 * np.pi * frequency * t))
        self.alarm_sound = pygame.mixer.Sound(buffer=buf)
   
    def start_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_thread = threading.Thread(target=self._alarm_loop)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
   
    def stop_alarm(self):
        self.alarm_active = False
        pygame.mixer.stop()
        if self.alarm_thread:
            self.alarm_thread.join(timeout=1.0)
            self.alarm_thread = None
   
    def _alarm_loop(self):
        while self.alarm_active:
            self.alarm_sound.play(loops=-1)  # Loop indefinitely
            time.sleep(0.1)  # Small delay to prevent CPU overload