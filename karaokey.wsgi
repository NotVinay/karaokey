#!/usr/bin/python3.6

import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/karaokey/")

from karaokey import app as application
application.secret_key = 'Temp Key'
