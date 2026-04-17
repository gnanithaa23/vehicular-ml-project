from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.layers import Dense, Dropout

import keras.layers
from sklearn.ensemble import RandomForestClassifier




global X, Y, dataset, X_train, X_test, y_train, y_test
global algorithms, accuracy, f1, precision, recall, classifier


def ProcessData(request):
    if request.method == 'GET':
        global X, Y, dataset, X_train, X_test, y_train, y_test
        dataset = pd.read_csv("Dataset/engine_data.csv")
        dataset.fillna(0, inplace=True)
        label = dataset.groupby('Engine Condition').size()
        columns = dataset.columns
        temp = dataset.values
        dataset = dataset.values
        X = dataset[:, 0:6]
        Y = dataset[:, -1]  # the last column

        Y = Y.astype(int)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Apply SMOTE to handle class imbalance
        

        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th>" + font + columns[i] + "</th>"
        output += "</tr>"
        for i in range(len(temp)):
            output += "<tr>"
            for j in range(0, temp.shape[1]):
                output += '<td><font size="" color="black">' + str(temp[i, j]) + '</td>'
            output += "</tr>"
        context = {'data': output}

        label.plot(kind="bar")
        plt.title("Engine quality Graph, 0 (BAD quality) & 1 (GOOD Quality)")
        plt.show()

        return render(request, 'UserScreen.html', context)


        

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from django.shortcuts import render

def TrainRF(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall, classifier
    if request.method == 'GET':
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        # Define model path
        os.makedirs('model', exist_ok=True)
        model_path = 'model/RandomForestClassifier.pkl'

        if os.path.exists(model_path):
            # Load the trained model
            rf = joblib.load(model_path)
            print("Model loaded successfully.")
        else:
            # Train the model and save it
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            joblib.dump(rf, model_path)
            print("Model saved successfully.")

        # Predict and evaluate
        y_pred = rf.predict(x_test)
        p = precision_score(y_test, y_pred, average='macro') * 100
        r = recall_score(y_test, y_pred, average='macro') * 100
        f = f1_score(y_test, y_pred, average='macro') * 100
        a = accuracy_score(y_test, y_pred) * 100

        # Save classifier globally
        classifier = rf

        # Append metrics
        algorithms.append("RandomForestClassifier")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)

        # Build HTML table for display
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>" + "".join(f"<th>{font}{col}</th>" for col in arr) + "</tr>"
        for i in range(len(algorithms)):
            output += (
                "<tr>"
                f"<td>{font}{algorithms[i]}</td>"
                f"<td>{font}{accuracy[i]}</td>"
                f"<td>{font}{precision[i]}</td>"
                f"<td>{font}{recall[i]}</td>"
                f"<td>{font}{fscore[i]}</td>"
                "</tr>"
            )

        context = {'data': output}
        return render(request, 'UserScreen.html', context)


import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from django.shortcuts import render

def TrainEnsembleModel(request):
    if request.method == 'GET':
        global X, Y
        global algorithms, accuracy, fscore, precision, recall

        # Reset lists
        algorithms = []
        accuracy = []
        fscore = []
        precision = []
        recall = []

        # Prepare data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        num_classes = len(np.unique(Y))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        # Create model directory
        os.makedirs('models', exist_ok=True)
        model1_path = 'models/model1.h5'
        model2_path = 'models/model2.h5'

        # Load or train model 1
        if os.path.exists(model1_path):
            model1 = load_model(model1_path)
            print("Model 1 loaded successfully.")
        else:
            model1 = Sequential([
                Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
                Dense(num_classes, activation='softmax')
            ])
            model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model1.fit(x_train, y_train_cat, epochs=30, verbose=1)
            model1.save(model1_path)
            print("Model 1 saved successfully.")

        # Load or train model 2
        if os.path.exists(model2_path):
            model2 = load_model(model2_path)
            print("Model 2 loaded successfully.")
        else:
            model2 = Sequential([
                Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model2.fit(x_train, y_train_cat, epochs=30, verbose=1)
            model2.save(model2_path)
            print("Model 2 saved successfully.")

        # Predict and ensemble (soft voting)
        pred1 = model1.predict(x_test)
        pred2 = model2.predict(x_test)
        ensemble_pred = (pred1 + pred2) / 2
        y_pred = np.argmax(ensemble_pred, axis=1)
        y_true = y_test

        # Calculate metrics
        p = precision_score(y_true, y_pred, average='macro') * 100
        r = recall_score(y_true, y_pred, average='macro') * 100
        f = f1_score(y_true, y_pred, average='macro') * 100
        a = accuracy_score(y_true, y_pred) * 100

        # Append to global lists
        algorithms.append("Ensemble Deep Learning")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)

        # Build HTML table
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>" + "".join(f"<th>{font}{col}</th>" for col in arr) + "</tr>"
        for i in range(len(algorithms)):
            output += (
                f"<tr><td>{font}{algorithms[i]}</td>"
                f"<td>{font}{accuracy[i]}</td>"
                f"<td>{font}{precision[i]}</td>"
                f"<td>{font}{recall[i]}</td>"
                f"<td>{font}{fscore[i]}</td></tr>"
            )

        context = {'data': output}
        return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global rf  # assume rf is your trained model loaded globally or within this function

        # Load the test data
        test = pd.read_csv("Dataset/test.csv")
        test.fillna(0, inplace=True)
        test_values = test.values  # Convert to NumPy array for prediction

        # Load the trained Random Forest model
        import os
        import joblib
        rf_model_path = 'model/RandomForestClassifier.pkl'
        if os.path.exists(rf_model_path):
            rf = joblib.load(rf_model_path)
            print("Random Forest model loaded successfully.")
        else:
            return render(request, 'UserScreen.html', {'data': 'Model not found!'})

        # Predict using the model
        y_pred = rf.predict(test_values)

        # Map predicted numeric labels to meaningful string labels
        label_map = {0: 'Bad Engine Health', 1: 'Good Engine Health'}

        # Create HTML table to display the results
        arr = ['Test Data (Features)', 'Vehicular Engine Health Prediction']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for col in arr:
            output += "<th>" + font + col + "</th>"
        output += "</tr>"

        for i in range(len(y_pred)):
            test_features = ", ".join([str(x) for x in test_values[i]])
            label = label_map[y_pred[i]]
            output += f"<tr><td>{font}{test_features}</td><td>{font}{label}</td></tr>"

        context = {'data': output}
        return render(request, 'UserScreen.html', context)



def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Vehicular',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed. Please retry'}
            return render(request, 'UserLogin.html', context)        

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Vehicular',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Vehicular',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      


