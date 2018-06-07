clc; close all; clear all;
w=10; d=5; iters=200;
dataName='optdigits_train';
dataTest='optdigits_test';

[fig1,fig2]=neuralNet(d,w,dataName,dataTest,iters);