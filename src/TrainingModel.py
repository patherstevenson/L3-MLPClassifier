#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`DataLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: march 2022

TrainingModel Module
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def train(mlp, dataloader, categories):
  """

  Function that create a LinearDiscriminantAnalysis (LDA) fited with the extracted features from the train section of the given dataloader.
  Then the given MLP is fited with the transformed features list returns by the LDA created.

  Finaly we get the accuracy from the method score of MLP. We do this for both section training and validation

  :param mlp: a classifier multilayer perceptron to use
  :param dataloader: the DataLoader used during the preprocessing dataset loading
  :param categories: the list of data categories
  :type mlp: MLPClassifier
  :type dataloader: DataLoader
  :type categories: List

  :return: (the train accuracy, the validation accuracy, the used LinearDiscriminantAnalysis object during training)
  :rtype: tuple
  """
  dl = dataloader

  l_features, l_cat = dl.convergeFeatures(categories, 'train')
  
  lda = LinearDiscriminantAnalysis(n_components=1).fit(l_features, l_cat)
  
  l_features = lda.transform(l_features)

  mlp.fit(l_features, l_cat)

  train_accuracy = mlp.score(l_features, l_cat)

  l_features, l_cat = dl.convergeFeatures(categories, 'valid')
  l_features = lda.transform(l_features)

  valid_accuracy = mlp.score(l_features, l_cat)
  
  return train_accuracy, valid_accuracy, lda
