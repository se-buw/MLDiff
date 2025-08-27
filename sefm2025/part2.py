from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, pairwise_distances 
from sklearn.decomposition import PCA

import numpy as np
import os


from z3 import *
import mldiff.logReg2smtR as logReg2smt
import mldiff.mlp2smtR as mlp2smt

import joblib
# As another example, a team of engineers is working on a facial recognition system. 
# Maintaining the existing Neural Network (Multi-Layer Perceptron) classifier has
# become costly as the number of target faces grows (classes: employee IDs). 
# As an alternative, the team investigates a binary Logistic Regression classifier (classes: access/no access). 
# The old MLP classifier is trained to classify employees 0-9 while the candidate 
# Logistic Regression classifier is trained on more employees (0-15) where 0-5 have access and 6-15 do not. 
# One question of the engineers is whether the new LogReg classifier would give access to any 
# employee with an ID other than 0-5 when checked with the existing and trusted MLP classifier.


def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()   
    yield from all_smt_rec(list(initial_terms))

# use the trained olivetty faces MLP here

faces_mlp = joblib.load('models\olivetti_mlp_20_withPCA.joblib')
faces_pca = joblib.load('models\olivetti_pca.joblib')
faces_logReg = joblib.load('models\olivetti_logReg_example.joblib')


faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# pca = PCA(n_components=24)
# pca.fit(X)

X = faces_pca.transform(X)

X = X[y < 16]
y = y[y < 16]
# train a new LogReg classifier on the extended dataset of faces 0-15 with two classes where faces 0-5 have class=1 and faces 6-15 have class = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_logReg = y_train.copy()
y_train_logReg[y_train_logReg < 6] = 1
y_train_logReg[y_train_logReg >= 6] = 0
y_test_logReg = y_test.copy()
y_test_logReg[y_test_logReg < 6] = 1
y_test_logReg[y_test_logReg >= 6] = 0

# logReg = LogisticRegression(max_iter=1000)
# logReg.fit(X_train, y_train_logReg)
# print("LogReg f1-score:", f1_score(y_test_logReg, logReg.predict(X_test), average='macro'))

# # save a good model of the LogReg classifier
# joblib.dump(logReg, 'models/olivetti_logReg_example.joblib')


# convert the two classifiers to SMT
clMlp = Int("classMLP")
s = mlp2smt.toSMT(faces_mlp, str(clMlp), Solver())

clLogReg = Int("classLogReg")
logReg2smt.toSMT(faces_logReg, str(clLogReg), s)


# use the feature constraints from the olivetty faces dataset (pixels between 0 and 1 or use the min and max values of face 6-9) 

# faces we look for
Xsearch = X[(y >= 6) & (y <= 9)]
print(Xsearch.shape)

features = []
for i in range(X.shape[1]):
    f = Real("x" + str(i))
    features.append(f)
    # minPca = faces_pca.transform(np.zeros((1, 4096)))[0]
    # print(minPca)
    # maxPca = faces_pca.transform(np.ones((1, 4096)))[0]
    # print(maxPca)
    # for i in range(len(features)):
    #     lower = min(minPca[i], maxPca[i])
    #     upper = max(minPca[i], maxPca[i])
    #     s.add(features[i] >= lower, features[i] <= upper)
    s.add(f >= min(Xsearch[:, i]), f <= max(Xsearch[:, i]))

# run the query classLogReg == 1, i.e., access, and classMLP > 5, i.e., no access
s.add(clLogReg == 1)
s.add(clMlp > 5)


# for solutions that differ in classMLP
if s.check() == sat:
    for m in all_smt(s, [clLogReg, clMlp]):
        face = []
        for f in features:
            face.append(float(m.eval(f, model_completion=True).as_fraction()))
        face = faces_pca.inverse_transform(face)
        face = face.reshape(64, 64)

        print("LogReg class:", m[clLogReg])
        print("MLP class:", m[clMlp])
        
        # save the face to disk as face_generated_[6..9].png
        save_dir = "sefm2025/results/part_2/euclidean/"
        plt.imsave(os.path.join(save_dir, "face_generated_" + str(m[clMlp]) + ".png"), face, cmap='gray')
        
        # calculate the closest face based on mean_square_error from the predicted class to visually check for likeness
        face_reshaped = face.reshape(1, -1)

        # find the distance to the closest face using the euclidean distance
        # distances = np.linalg.norm(faces.data - face_reshaped, axis=1)
        
        # find the distance to the closest face using the mean squared error
        distancesMLPPrediction = np.mean((faces.data[faces.target == m[clMlp]] - face_reshaped) ** 2, axis=1)
        distancesLogRegPrediction = np.mean((faces.data[faces.target <=5 ] - face_reshaped) ** 2, axis=1)
        
        # find the closest face
        closest_face_indexMLP = np.argmin(distancesMLPPrediction)
        closest_distanceMLP = distancesMLPPrediction[closest_face_indexMLP]
        
        closest_face_indexLogReg = np.argmin(distancesLogRegPrediction)
        closest_distanceLogReg = distancesLogRegPrediction[closest_face_indexLogReg]
        
        # save the two closest faces to disk as face_generated_[6..9]_close[1..2].png
        closest_faceMLP = faces.data[closest_face_indexMLP].reshape(64, 64)
        closest_faceLogReg = faces.data[closest_face_indexLogReg].reshape(64, 64)
        plt.imsave(os.path.join(save_dir, "face_generated_" + str(m[clMlp]) + "_close.png"), closest_faceMLP, cmap='gray')
        plt.imsave(os.path.join(save_dir, "face_generated_" + str(m[clMlp]) + "_access_close.png"), closest_faceLogReg, cmap='gray')
    
    # extract a model and show the corresponding faces (consider iterating over solutions to find a nice example)
    
    # save reverse transformed model as face_generated_[6..9].png

    
    # save the two closest faces to disk as face_generated_[6..9]_close[1..2].png

    