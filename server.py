import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/pawel/Programowanie/Python3/Zespol2/static/files' #TU ZMIENIC SCIEZKE DO FOLDERU GDZIE SIE UPLOADUJE
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Nie wybrano pliku')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('Nie wybrano pliku')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Wgraj plik z pismem</title>
    <h1>Wgraj plik z pismem</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Wgraj!>
    </form>
    '''

@app.route('/uploaded_file')
def uploaded_file():
    filename = request.args['filename']
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return '''
    GRATULACJE!
    '''


if __name__ == '__main__':
    app.run()
