from flask import Flask, flash, render_template, request, jsonify, session, send_file, redirect
import os
from flask_sqlalchemy import SQLAlchemy

# setting up the flask app
app = Flask(__name__)
# importing the app configurations
app.config.from_pyfile('config.py')

# setting up remote database
db = SQLAlchemy(app)

page = 'home'

# app imports after initialising the app and db
from .controllers import file_actions, predict
from application.model import Track, User, Rating

# create database tables only for feedback forms
db.create_all()

@app.route('/')
def home_page():
    """
        Renders home page
    """
    return render_template('home.html')

@app.route('/results')
def results_page():
    """
        Renders Results of Process page
    """
    page = 'results'
    return render_template('results.html')

@app.route('/faq')
def help_page():
    """
        Renders faq page
    """
    page = 'help'
    return render_template('faq.html')

@app.route('/about')
def about_page():
    """
        Renders about page
    """
    page = 'about'
    return render_template('about.html')

@app.route('/research')
def research_study_page():
    """
        Renders about page
    """
    page = 'research'
    dir_path = os.path.join(app.config['ROOT'], 'static/cognitive_research')
    _, folders, _ = next(os.walk(dir_path))
    return render_template('research.html', track_dirs=folders)

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
        #flash("Saving uploaded file")
        saved = file_actions.save_file()
        if saved:
            #flash("Uploaded file saved")
            dir_path = os.path.join(app.config['AUDIO_DIR'], session['token'])
            #flash("Predicting...")
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
    file
        <file> if requested file exist and is available to the user, <dict> otherwise
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


@app.route('/submit-research', methods=['POST'])
def submitResearch():
    """
    Submits the research form
    """
    form_data = request.form
    user = User(feedback=form_data['feedback'])
    db.session.add(user)
    for i in range(1, 30):
        s_i = str(i)
        if "title_"+s_i in form_data:
            track = Track.query.filter_by(title=request.form["title_"+s_i]).first()
            db.session.add(Rating(user_id=user.id,
                                  track_id=track.id,
                                  rating=form_data["rating_"+s_i]))
    try:
        db.session.commit()
        return jsonify({'success': True})
    except:
        return jsonify({'success': False})


@app.route('/clear-session', methods=['GET'])
def clearSession():
    """
    Clears the session
    """
    session.clear()
    return redirect(url_for('home_page'))

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