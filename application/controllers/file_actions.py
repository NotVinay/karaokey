import os, shutil
import soundfile as sf
import secrets
from flask import request, session
from flask import current_app as app


def supported_file(filename):
    name, ext = os.path.splitext(filename)
    return '.' in filename and ext.lower() in app.config['SUPPORTED_EXTENSIONS']


def create_unique_dir():
    generated = None

    while not generated:
        rnd_token = secrets.token_hex(16)
        new_dir = os.path.join(app.config['AUDIO_DIR'], rnd_token)
        token_is_unique = False if os.path.exists(new_dir) and os.path.isdir() else True
        if token_is_unique:
            session['token'] = rnd_token
            os.mkdir(new_dir, mode=0o700)
            generated = True
        else:
            generated = False

    return new_dir


def save_file():
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
        data, sr = sf.read(temp_path)

        sf.write(new_file_path, data, sr)
        return True
    else:
        return False


def remove_session_token():
    if session['token'] is not None:
        session_dir = os.path.join(app.config['AUDIO_DIR'], session['token'])
        # remove the directory of that token
        shutil(session_dir)
        session.pop('token', None)
        return True
    elif session['token'] is None:
        return False


def get_file_path():
    file_name = request.args.get('file_name')

    if not file_name or file_name is None:
        return {'error': True, 'description': "Error in request parameters"}
    if 'token' in session:
        if session['token'] and session['token'] is not None:
            dir_path = os.path.join(app.config['AUDIO_DIR'], session['token'])
            file_path = os.path.join(dir_path, file_name)
            if os.path.isdir(dir_path) and os.path.isfile(file_path):
                return file_path
            else:
                return {'error': True, 'description': "Error the file doesn't exist anymore"}

    return {'error': True, 'description': "Invalid Request"}






