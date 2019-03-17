"""
Contains the configuration of the Flask app and other constants
"""
import os
from datetime import timedelta

DEBUG = True
TESTING = True
SESSION_COOKIE_NAME = 'session'
PERMANENT_SESSION_LIFETIME = timedelta(days=2)
SECRET_KEY = "b'\x89b=|\xbb\x99\x0e\x0f\xd4\xa7\x06\x90\xd0\x80&j'"

ROOT = os.path.dirname(os.path.realpath(__file__))
AUDIO_DIR = os.path.join(ROOT, 'audio/')
MODEL_PATH = os.path.join(ROOT, 'controllers/models/test2_2019-03-15_14-41_LSTM_accompaniment_B16_H256_S1000.pt')

SUPPORTED_EXTENSIONS = [".aac",
                        ".aif",
                        ".au",
                        ".flac",
                        ".gsm",
                        ".mp3",
                        ".ogg",
                        ".ra",
                        ".wav",
                        ".wma",
                        ".wav"]
