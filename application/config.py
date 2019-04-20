"""
Contains the configuration of the Flask app and other constants
"""
import os
from datetime import timedelta

# Only for Development Stage
DEBUG = True
TESTING = True

# setting session
SESSION_COOKIE_NAME = 'session'
PERMANENT_SESSION_LIFETIME = timedelta(days=1) # session expiry duration
SECRET_KEY = "b'\x89b=|\xbb\x99\x0e\x0f\xd4\xa7\x06\x90\xd0\x80&j'"

# MySQL setup for remote server only
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://vinay:Vi^ay12345@10.0.4.71/karaokey'

# MySQL setup for uni environment server
# SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://w1572032:AH1JedDWtkNz@elephant.ecs.westminster.ac.uk/w1572032_0'

# Global app configs
ROOT = os.path.dirname(os.path.realpath(__file__))
AUDIO_DIR = os.path.join(ROOT, 'audio/')
MODEL_PATH = os.path.join(ROOT, 'controllers/models/30_2019-04-07_11-49_Generalised_Recurrent_Model_relu_accompaniment_B16_H512_S5000_adam.pt')
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
                        ".wav",
                        ".m4a"]
