import os
from datetime import timedelta

DEBUG = True
TESTING = True
SESSION_COOKIE_NAME = 'session'
PERMANENT_SESSION_LIFETIME = timedelta(days=1)
SECRET_KEY = "b'\x89b=|\xbb\x99\x0e\x0f\xd4\xa7\x06\x90\xd0\x80&j'"

ROOT = os.path.dirname(os.path.realpath(__file__))
AUDIO_DIR = os.path.join(ROOT, 'audio/')
SUPPORTED_EXTENSIONS = set([".aif",
                            ".au",
                            ".flac",
                            ".ogg",
                            ".wav"])

# ALL_EXTS = [".aac",
#         ".aif",
#         ".au",
#         ".flac",
#         ".gsm",
#         ".mp3",
#         ".ogg",
#         ".ra",
#         ".raw",
#         ".vox",
#         ".wav",
#         ".wma",
#         ".wav"]
