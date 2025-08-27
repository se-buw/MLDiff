#!/bin/bash
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=dt -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=dt -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=dt -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=dt -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=svm -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=svm -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=svm -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=svm -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=logreg -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=logreg -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=logreg -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=logreg -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=mlp -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=mlp -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=mlp -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=iris -c1=mlp -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=dt -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=dt -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=dt -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=dt -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=svm -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=svm -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=svm -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=svm -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=logreg -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=logreg -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=logreg -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=logreg -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=mlp -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=mlp -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=mlp -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=digits -c1=mlp -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=dt -c2=dt -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=dt -c2=svm -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=dt -c2=logreg -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=dt -c2=mlp -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=svm -c2=dt -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=svm -c2=svm -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=svm -c2=logreg -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=svm -c2=mlp -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=logreg -c2=dt -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=logreg -c2=svm -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=logreg -c2=logreg -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=logreg -c2=mlp -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=mlp -c2=dt -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=mlp -c2=svm -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=mlp -c2=logreg -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=olivetti -c1=mlp -c2=mlp -pca -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=dt -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=dt -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=dt -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=dt -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=svm -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=svm -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=svm -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=svm -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=logreg -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=logreg -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=logreg -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=logreg -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=mlp -c2=dt  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=mlp -c2=svm  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=mlp -c2=logreg  -ed 0 -ea 0 -fc -sd=RQ3 
timeout -k 0s 60m python -m mldiff.diff -d=cancer -c1=mlp -c2=mlp  -ed 0 -ea 0 -fc -sd=RQ3 
