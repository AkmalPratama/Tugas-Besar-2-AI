from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
  #IMPORT FILE CSV
  import pandas as pd
  from sklearn import preprocessing
  attributeName = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation","relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]
  data=pd.read_csv("D:/CencusIncome.data.txt", names=attributeName)
	# GANTI PATHNYA ^^^^^^^^^^^^^^^^^^^^^

  capitalgainmean = data["capital-gain"].mean()
  capitallossmean = data["capital-loss"].mean()
  hoursmean = data["hours-per-week"].mean()
  workclassmode = data["workclass"].mode()[0]
  educationmode = data["education"].mode()[0]
  maritalmode = data["marital-status"].mode()[0]
  occupationmode = data["occupation"].mode()[0]
  relationshipmode = data["relationship"].mode()[0]
  racemode = data["race"].mode()[0]
  sexmode = data["sex"].mode()[0]

  data["capital-gain"] = data[["capital-gain"]].replace(' ?', data["capital-gain"].mean())
  data["capital-loss"] = data[["capital-loss"]].replace(' ?', data["capital-loss"].mean())
  data["hours-per-week"] = data[["hours-per-week"]].replace(' ? ', data["hours-per-week"].mean())

  data["workclass"] = data[["workclass"]].replace(' ?', " " + data["workclass"].mode()[0])
  data["education"] = data[["education"]].replace(' ?', " " + data["education"].mode()[0])
  data["marital-status"] = data[["marital-status"]].replace(' ?', " " + data["marital-status"].mode()[0])
  data["occupation"] = data[["occupation"]].replace(' ?', " " + data["occupation"].mode()[0])
  data["relationship"] = data[["relationship"]].replace(' ?', " " + data["relationship"].mode()[0])
  data["race"] = data[["race"]].replace(' ?', " " + data["race"].mode()[0])
  data["sex"] = data[["sex"]].replace(' ?', " " + data["sex"].mode()[0])



  cidata = data.as_matrix()

  le1 = preprocessing.LabelEncoder()
  le1.fit(cidata[:,1])
  list(le1.classes_)
  cidata[:,1] = le1.transform(cidata[:,1])


  le3 = preprocessing.LabelEncoder()
  le3.fit(cidata[:,3])
  list(le3.classes_)
  cidata[:,3] = le3.transform(cidata[:,3])

  le5 = preprocessing.LabelEncoder()
  le5.fit(cidata[:,5])
  list(le5.classes_)
  cidata[:,5] = le5.transform(cidata[:,5])

  le6 = preprocessing.LabelEncoder()
  le6.fit(cidata[:,6])
  list(le6.classes_)
  cidata[:,6] = le6.transform(cidata[:,6])

  le7 = preprocessing.LabelEncoder()
  le7.fit(cidata[:,7])
  list(le7.classes_)
  cidata[:,7] = le7.transform(cidata[:,7])

  le8 = preprocessing.LabelEncoder()
  le8.fit(cidata[:,8])
  list(le8.classes_)
  cidata[:,8] = le8.transform(cidata[:,8])

  le9 = preprocessing.LabelEncoder()
  le9.fit(cidata[:,9])
  list(le9.classes_)
  cidata[:,9] = le9.transform(cidata[:,9])

  le13 = preprocessing.LabelEncoder()
  le13.fit(cidata[:,13])
  list(le13.classes_)
  cidata[:,13] = le13.transform(cidata[:,13])

  le14 = preprocessing.LabelEncoder()
  le14.fit(cidata[:,14])
  list(le14.classes_)
  cidata[:,14] = le14.transform(cidata[:,14])

  workclass = ' '+request.form['workclass']
  education = ' '+request.form['education']
  maritalstatus = ' '+request.form['marital-status']
  occupation = ' '+request.form['occupation']
  relationship = ' '+request.form['relationship']
  race = ' '+request.form['race']
  sex = ' '+request.form['sex']
  print("capital gain =" + request.form['capital-gain'] + "....")
  if request.form['capital-gain']=="":
    capitalgain = ' '+str(capitalgainmean)
  else:
    capitalgain = ' '+request.form['capital-gain']

  if request.form['capital-loss']=="":
    capitalloss = ' '+str(capitallossmean)
  else:
    capitalloss = ' '+request.form['capital-loss']

  if request.form['hours-per-week']=="":
    hoursperweek = ' '+str(hoursmean)
  else:
    hoursperweek = ' '+request.form['hours-per-week']

  
  appended = workclass+','+education+','+maritalstatus+','+occupation+','+relationship+','+race+','+sex+','+capitalgain+','+capitalloss+','+hoursperweek

  import sys
  if sys.version_info[0] < 3:
    from StringIO import StringIO
  else:
    from io import StringIO
  TESTDATA=StringIO(appended)
  attributeName = ["workclass", "education", "marital-status", "occupation","relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"]
  data = pd.read_csv(TESTDATA, names=attributeName)

  data["workclass"] = data[["workclass"]].replace(' ?', workclassmode)
  data["education"] = data[["education"]].replace(' ?',  educationmode)
  data["marital-status"] = data[["marital-status"]].replace(' ?', maritalmode)
  data["occupation"] = data[["occupation"]].replace(' ?', occupationmode)
  data["relationship"] = data[["relationship"]].replace(' ?',  relationshipmode)
  data["race"] = data[["race"]].replace(' ?',  racemode)
  data["sex"] = data[["sex"]].replace(' ?', sexmode)


  cidata = data.as_matrix()
  cidata[:,0] = le1.transform(cidata[:,0])

  cidata[:,1] = le3.transform(cidata[:,1])

  cidata[:,2] = le5.transform(cidata[:,2])

  cidata[:,3] = le6.transform(cidata[:,3])

  cidata[:,4] = le7.transform(cidata[:,4])

  cidata[:,5] = le8.transform(cidata[:,5])

  cidata[:,6] = le9.transform(cidata[:,6])

  from sklearn.externals import joblib
  filename = 'clf.sav'
  #Load model
  loaded_model = joblib.load(filename)
  pred = loaded_model.predict(cidata)
  predict = le14.inverse_transform(pred)
  predstring = predict[0]
  return render_template("index.html",result = predstring)


if __name__ == '__main__':
   app.run(debug = True)