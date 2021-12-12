from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

# load the model from disk
filename = 'mnb.pkl'
mnb = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tfidf.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		input_title = request.form['input_title']
		data = [input_title]
		vect = cv.transform(data).toarray()
		my_prediction = mnb.predict(vect)
		# my_prediction = mnb.predict_proba(vect)
 		# output='{0:.{1}f}'.format(my_prediction[0][1], 2)

    	# if output>str(0.5):
    	#     return render_template('home.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    	# else:
        # 	return render_template('home.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
	return render_template('result.html', prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)