from flask import Flask, flash, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import os
from .controllers import file_actions, predict

app = Flask(__name__)
app.config.from_pyfile('config.py')

page = 'home'

@app.route('/')
def home_page():
    """
        Renders home page
    """
    page = 'home'
    return render_template('home.html')


@app.route('/karaokey')
def karaokey_page():
    """
        Renders Karaokey application page
    """
    page = 'karaokey'
    return render_template('karaokey.html')


@app.route('/help')
def help_page():
    """
        Renders help page
    """
    page = 'help'
    return render_template('help.html')


@app.route('/about')
def about_page():
    """
        Renders about page
    """
    page = 'about'
    return render_template('about.html')

"""
    ------------------------------
            GETS AND POSTS
    ------------------------------
"""


@app.route('/separate', methods=['GET', 'POST'])
def separate():
    if 'music' not in request.files:
        flash('No Music File')
        return 'No Music File'
    audio_file = request.files['music']

    # if no file is selected

    if audio_file.filename == '':
        flash('No selected file')
        return 'No File Selected'

    if audio_file and not file_actions.supported_file(audio_file.filename):
        flash('File Not Supported')
        return 'File Not Supported'

    elif audio_file and file_actions.supported_file(audio_file.filename):
        saved = file_actions.save_file()
        if saved:
            # TODO : Call Predict Function
            flash('Saved File')
        else:
            flash('Error in saving the file')
            return 'Error in saving the file'

    return jsonify(session['token'])


@app.route('/download', methods=['GET'])
def download():
    ret = None

    req_file = request.args.get('file_name')
    if req_file and req_file is not None:
        file_path = file_actions.get_file_path()
        if file_path:
            try:
                return send_file(file_path, attachment_filename=req_file)
            except Exception as e:
                return jsonify(e)
    else:
        ret = {'error': True, 'description': "Error in request parameters"}

    return jsonify(ret)


"""
    ------------------------------
            ERROR HANDLERS
    ------------------------------
"""


@app.errorhandler(404)
def error_404(error):
    """
        Renders 404 - Page not found error page
    """
    return render_template('error_404.html')


@app.errorhandler(405)
def error_405(error):
    """
        Renders 405 - Method Not Allowed Error page
    """
    return render_template('error_405.html')


@app.errorhandler(500)
def error_500(error):
    """
        Renders 500 - Internal Server Error page
    """
    return render_template('error_500.html')