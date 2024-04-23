from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
import pickle
import cv2
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
import xgboost as xg 

global filename, main
global precision, recall, fscore, accuracy, pathlabel, selector, pso
global dataset
global X_train, X_test, y_train, y_test, text, X1
global xg_model, labels
classifier = RandomForestClassifier()
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

def uploadDataset(): #function to upload dataset
    global filename, dataset, labels, X, Y
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)    
    labels = ['Anger', 'Disgust', 'Happy']
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j], 0)
                    img = cv2.resize(img, (28, 28))
                    X.append(img.ravel())
                    label = -1
                    if 'anger' in name:
                        label = 0
                    if 'disgust' in name:
                        label = 1
                    if 'happy' in name:
                        label = 2
                    Y.append(label)
                    print(name+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        print(Y)
        print(Y.shape)
        print(np.unique(Y, return_counts=True))
        np.save('model/X',X)
        np.save('model/Y',Y)
    unique, count = np.unique(Y, return_counts=True)
    text.insert(END,"Total Casme Micro Expression images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Class Labels found in Casme Dataset : "+str(labels))
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (6, 4)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks()
    plt.tight_layout()
    plt.show()

def preprocessDataset():
    global dataset, X, Y, X1
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X1 = X
    text.insert(END,"Dataset Normalization & Shuffling Process Completed")  

def testSplit():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Total images found in dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each image = "+str(X.shape[1])+"\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

def runGA():
    global X, Y, selector
    if os.path.exists("model/ga.npy"):
        selector = np.load("model/ga.npy")
    else:
        estimator = RandomForestClassifier()
        #defining genetic alorithm object
        selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=10, n_population=10, crossover_proba=0.5, mutation_proba=0.2,
                                      n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=10,
                                      caching=True, n_jobs=-1)
        ga_selector = selector.fit(X, Y) #train with GA weights
        selector = ga_selector.support_
        np.save("model/ga", selector)
    X = X[:,selector] #extract selected features
    return X, Y, selector

#PSO function
def f_per_particle(m, alpha):
    global X, Y, classifier
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    classifier.fit(X_subset, Y)
    P = (classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def runPSO():
    global X, Y, pso
    if os.path.exists("model/pso.npy"):
        pso = np.load("model/pso.npy")
    else:
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
        dimensions = X.shape[1] # dimensions should be the number of features
        optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
        cost, pso = optimizer.optimize(f, iters=35)#OPTIMIZING FEATURES
        np.save("model/pso", pso)
    X = X[:,pso==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1

def runHybrid():
    global X, Y
    text.delete('1.0', END)
    text.insert(END,"Total features found in each Facial Landmark before applying GA : "+str(X.shape[1])+"\n")
    runGA()
    text.insert(END,"Total features found in each Facial Landmark after applying GA : "+str(X.shape[1])+"\n")
    text.insert(END,"Total features found in each Facial Landmark before applying PSO : "+str(X.shape[1])+"\n")
    runPSO()
    text.insert(END,"Total features found in each Facial Landmark after applying PSO : "+str(X.shape[1])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(6, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()    
    
def runImprovedXgboost():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, xg_model
    global precision, recall, fscore, accuracy
    precision = []
    recall = []
    fscore = []
    accuracy = []
    xg_model = xg.XGBClassifier(learning_rate=0.08,   # Learning rate (default=0.1)
                                 max_depth=4,        # Maximum depth of a tree (default=3)
                                 reg_alpha=0.1,      # L1 regularization term on weights (default=0)
                                 reg_lambda=1.1)     # L2 regularization term on weights (default=1.0))
    xg_model.fit(X_train, y_train)
    predict = xg_model.predict(X_test)
    calculateMetrics("Hybrid Improved XGBoost", y_test, predict) 

def close():
    main.destroy()
    
def predict():
    global selector, pso, xg_model
    filename = filedialog.askopenfilename(initialdir="Videos")
    cap = cv2.VideoCapture(filename)
    expression_count = {"Angry": 0, "Disgust": 0, "Happy": 0}  
    while True:
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                x1 = x + 20
                y1 = y + 20
                w1 = w - 50
                h1 = h - 40
                roi = gray[y1:y1 + h1, x1:x1 + w1]
                cv2.imwrite("D:/MicroExpression/test1.jpg", roi)
                img = cv2.resize(roi, (28, 28))
                temp = []
                temp.append(img.ravel())
                temp = np.asarray(temp)
                temp = temp.astype('float32')
                temp = temp/255
                temp = temp[:, selector]  # extract selected features
                temp = temp[:, pso == 1]
                predict = xg_model.predict(temp)[0]
                output = ""
                if predict == 0:
                    output = "Angry"
                elif predict == 1:
                    output = "Disgust"
                elif predict == 2:
                    output = "Happy"
                expression_count[output] += 1
                cv2.putText(frame, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Video Output', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    plt.figure(figsize=(6, 4))
    plt.bar(expression_count.keys(), expression_count.values())
    plt.xlabel("Expressions")
    plt.ylabel("Frequency")
    plt.title("Expression Frequency in Uploaded Video")
    plt.show()    


# def predict():
#     global selector, pso, xg_model
#     filename = filedialog.askopenfilename(initialdir="Videos")
#     cap = cv2.VideoCapture(filename)
#     while True:
#         ret, frame = cap.read()
#         if ret == True:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#             for (x, y, w, h) in faces:
#                 x1 = x + 20
#                 y1 = y + 20
#                 w1 = w - 50
#                 h1 = h - 40
#                 roi = gray[y1:y1 + h1, x1:x1 + w1]
#                 cv2.imwrite("test1.jpg", roi)
#                 img = cv2.resize(roi, (28, 28))
#                 temp = []
#                 temp.append(img.ravel())
#                 temp = np.asarray(temp)
#                 temp = temp.astype('float32')
#                 temp = temp/255
#                 temp = temp[:, selector]  # extract selected features
#                 temp = temp[:, pso == 1]
#                 predict = xg_model.predict(temp)[0]
#                 output = ""
#                 if predict == 0:
#                     output = "Angry"
#                 elif predict == 1:
#                     output = "Disgust"
#                 elif predict == 2:
#                     output = "Happy"
#                 cv2.putText(frame, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.imshow('Video Output', frame)
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
    
def gui():
    global text, pathlabel, main
    main = tkinter.Tk()
    main.title("An Improved XGBoost Classifier for Micro Expression Recognition using Hybrid Optimization Algorithm") #designing main screen
    main.geometry("1300x1200")

    font = ('times', 16, 'bold')
    title = Label(main, text='An Improved XGBoost Classifier for Micro Expression Recognition using Hybrid Optimization Algorithm')
    title.config(bg='brown', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload Casme Dataset", command=uploadDataset)
    uploadButton.place(x=50,y=100)
    uploadButton.config(font=font1)  

    pathlabel = Label(main)
    pathlabel.config(bg='brown', fg='white')  
    pathlabel.config(font=font1)           
    pathlabel.place(x=360,y=100)

    preprocessButton = Button(main, text="Extract & Preprocess Features", command=preprocessDataset)
    preprocessButton.place(x=50,y=150)
    preprocessButton.config(font=font1) 

    splitButton = Button(main, text="Run Hybrid GA & PSO Optimization", command=runHybrid)
    splitButton.place(x=330,y=150)
    splitButton.config(font=font1) 

    hybridButton = Button(main, text="Train & Test Split", command=testSplit)
    hybridButton.place(x=650,y=150)
    hybridButton.config(font=font1) 

    xgboostButton = Button(main, text="Run Improved XGBoost", command=runImprovedXgboost)
    xgboostButton.place(x=50,y=200)
    xgboostButton.config(font=font1)

    predictButton = Button(main, text="Predict Expression from Video", command=predict)
    predictButton.place(x=330,y=200)
    predictButton.config(font=font1)

    exitButton = Button(main, text="Exit", command=close)
    exitButton.place(x=650,y=200)
    exitButton.config(font=font1)


    font1 = ('times', 12, 'bold')
    text=Text(main,height=19,width=150)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=250)
    text.config(font=font1)
    main.config(bg='brown')
    main.mainloop()

if __name__ == "__main__":
    gui()




