from gender import Gender
from pos_tagger import POS
from flask import Flask, render_template, request
nltk.data.path.append('./nltk_data/')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gender', methods=['POST'])
def gender():
    name = request.form['name']
    test = Gender(name)
    answer = test.run()
    print answer
    return render_template('gender.html', data=answer)

@app.route('/pos', methods=['POST'])
def pos():
    sentence = request.form['sentence']
    index = int(request.form['index'])
    print type(index)
    test = POS()
    answer = test.run(sentence,index)
    print answer
    return render_template('pos.html', data=answer)

if __name__ == '__main__':
    app.run()
