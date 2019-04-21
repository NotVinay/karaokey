import os, shutil
import soundfile as sf
import librosa as lib
import secrets
from flask import request, session
from flask import current_app as app


__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Production"


def supported_file(filename):
    """
    Check if the file is supported by the application is not or not.

    Parameters
    ----------
    filename : str
        file name of the file

    Returns
    -------
    bool
        True if supported by application and False otherwise
    """
    name, ext = os.path.splitext(filename)
    return '.' in filename and ext.lower() in app.config['SUPPORTED_EXTENSIONS']


def create_unique_dir():
    """
    Creates unique directory for storing uploaded file and separated files.
    Returns
    -------
    str
        sends the directory path
    """
    generated = None
    while not generated:
        rnd_token = secrets.token_hex(16)
        new_dir = os.path.join(app.config['AUDIO_DIR'], rnd_token)
        token_is_unique = False if os.path.exists(new_dir) and os.path.isdir() else True
        if token_is_unique:
            # remove existing token if any
            if 'token' in session:
                remove_session_token()
            session['token'] = rnd_token
            os.mkdir(new_dir, mode=0o777)
            generated = True
        else:
            generated = False

    return new_dir


def save_file():
    """
    Saves the uploaded music file on the server so that it can be processed by the model.

    Returns
    -------
    bool
        True if successfully saved, False otherwise
    """
    new_dir = create_unique_dir()
    new_file_path = os.path.join(new_dir, 'mixture.wav')

    audio_file = request.files['music']
    name, ext = os.path.splitext(audio_file.filename)

    if ext == '.wav':
        audio_file.save(new_file_path)
        return True
    elif supported_file(audio_file.filename):
        # if file is not .wav than convert it to .wav

        # temporarily save the file on the system
        temp_path = os.path.join(new_dir, audio_file.filename)
        audio_file.save(temp_path)

        # read the file from the system
        data, sr = lib.load(temp_path, mono=False, sr=None)

        # write as wav
        sf.write(new_file_path, data.T, sr)
        return True
    else:
        return False


def remove_session_token():
    """
    Removes the `token` from the `session`
    and also removes the directory associated with this session.

    Returns
    -------
    bool
        True if successfully removed and False otherwise
    """
    if session['token'] is not None:
        # get directory associated with the token.
        session_dir = os.path.join(app.config['AUDIO_DIR'], session['token'])
        # remove the directory of that token
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        session.pop('token', None)
        return True
    else:
        # session token doesn't exist
        return False


def get_audio_file_path(file_name):
    """
    gets the absolute file path of the file by checking the available session token.

    Parameters
    ----------
    file_name : str
        name of the file to retrieve

    Returns
    -------
    str
        if file_path is valid and available for the user
    bool
        False if file_path is not valid
    """
    # check session token
    if 'token' in session:
        if session['token'] and session['token'] is not None:
            # build file path
            dir_path = os.path.join(app.config['AUDIO_DIR'], session['token'])
            file_path = os.path.join(dir_path, file_name)
            if os.path.isdir(dir_path) and os.path.isfile(file_path):
                # file exists
                return file_path
            else:
                # file path doesn't exist
                return False
    else:
        # session token doesn't exist
        return False






