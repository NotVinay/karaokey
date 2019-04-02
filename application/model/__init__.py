from application import db
import datetime

class Track(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), unique=True)
    sub_set = db.Column(db.String(40))
    ratings = db.relationship('Rating',
                              lazy=True,
                              backref=db.backref('track_rating', lazy='joined'))

    def __init__(self, title, sub_set):
        self.title = title
        self.sub_set = sub_set

    def __repr__(self):
        return '<Track %r>' % self.title


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feedback = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    ratings = db.relationship('Rating',
                              lazy='select',
                              backref=db.backref('rating_by', lazy='joined'))

    def __init__(self, feedback):
        self.feedback = feedback

    def __repr__(self):
        return '<User %r>' % self.username


class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    track_id = db.Column(db.Integer, db.ForeignKey('track.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)

    def __init__(self, user_id, track_id, rating):
        self.user_id = user_id
        self.track_id = track_id
        self.rating = rating

    def __repr__(self):
        return "<Rating %r" % self.id
