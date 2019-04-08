#!/usr/bin/python3.6

import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/karaokey/")

from application import app as application
application.secret_key = 'a3dcb4d229de6fde0db5686dee47145d'
