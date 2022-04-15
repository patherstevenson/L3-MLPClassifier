#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`MatrixLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: janvier 2022

Main Module

"""

from sklearn.neural_network import MLPClassifier
from CategoriesLoader import *
from MatrixLoader import *
from DataLoader import *
import TrainingModel
from HyperparameterTuning import *
from GenerateTest import *
from math import floor

def main():
  cl = CategoriesLoader("data/descriptions/categories.txt")
  cl.foundCategories()

  ml = MatrixLoader("data/train",cl.getCategories())
  ml.generateTrainAndValidMatrixImg()

  dl = DataLoader(ml.getDataImg(),ml.getDictionnaryEndIndex())
  dl.load()

  mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(32, 32), random_state=1,
      max_iter=10000, warm_start=True)

  train_accuracy, valid_accuracy, lda = TrainingModel.train(mlp, dl, cl.getCategories())
  print("\n\nWITHOUT Hyperparameter Tuning :\n\n\t- train_acc {}\n\t- val acc {}\n\t- rand {}".format(train_accuracy, valid_accuracy, 1/len(cl.getCategories())),end='\n\n')

  hpt = HyperparameterTuning(dl,cl.getCategories())

  hpt.startIterate()

  return dl, hpt, cl

if __name__ == "__main__":
  dl, hpt, cl = main()

  test = GenerateTest(dl,hpt,cl)

  hpt_hl_step = (hpt.max_layer // hpt.min_layer)

  nb_cat_slice = hpt_hl_step * floor(max(hpt.valid_accuracy)/hpt_hl_step)

  test.startTest(hpt_hl_step * nb_cat_slice)
