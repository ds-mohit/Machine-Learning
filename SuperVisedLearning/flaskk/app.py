from flask import Flask,render_template,request,url_for
app=Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route("/content")
def content():
    return render_template('content.html')


if __name__=='__main__':
    app.run(debug=True)