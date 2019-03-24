from flask import Flask, flash, render_template, request, jsonify, session, send_file
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

@app.route('/results')
def reults_page():
    """
        Renders Results of Process page
    """
    page = 'results'
    return render_template('results.html')

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
    """
    Handles the separation http request from user.

    Returns
    -------
    dict
        containing `token` for separation if separation is successful.
        or containing `error` if separation fails.
    """
    if 'music' not in request.files:
        return jsonify({'error': 'No Music File'})
    audio_file = request.files['music']

    # if no file is selected

    if audio_file.filename == '':
        return jsonify({'error': 'No File Selected'})

    if audio_file and not file_actions.supported_file(audio_file.filename):
        return jsonify({'error': 'File Not Supported'})

    elif audio_file and file_actions.supported_file(audio_file.filename):
        saved = file_actions.save_file()
        if saved:
            dir_path = os.path.join(app.config['AUDIO_DIR'], session['token'])
            predict.separate_file(dir_path)
            return jsonify({'token': session['token']})
        else:
            return jsonify({'error': 'Error in saving the file'})


@app.route('/download', methods=['GET'])
def download():
    """
    Sends requested file for the `token` in `session`.
    Token authentication ensures that file is only accessed by the user which it belongs to.
    Returns
    -------

    """
    ret = None
    req_file = request.args.get('file_name')
    if req_file and req_file is not None:
        file_path = file_actions.get_audio_file_path(req_file)
        if file_path:
            try:
                return send_file(file_path, attachment_filename=req_file)
            except Exception as e:
                return jsonify(e)
    else:
        ret = {'error': True, 'description': "Error in request parameters"}
    return jsonify(ret)

@app.route('/clear-session', methods=['GET'])
def clearSession():
    """
    Clears the session
    """
    pass
    # TODO ADD Clear Session Script


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