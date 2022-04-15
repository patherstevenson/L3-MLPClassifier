#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`FeatureLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: february 2022

FeatureLoader Module

"""

from sklearn.decomposition import PCA

class FeatureLoader:
  """
  Create a Featureloader that can return features from an image matrix by using a PCA
  """
  def __init__(self, features=32):
    """
    Linear dimensionality reduction using Singular Value Decomposition
    of the data to project it to a lower dimensional space. 
    The input data is centered but not scaled for each feature before applying the SVD.

    :param features: number of features
    :type features: int
    :build: a featureloader that can return features from an image matrix by using a PCA

    :UC: none
    """
    self.pca = PCA(n_components = features)


  def getPCA(self):
    """
    :return: the Principal Component Analysis (PCA) created at the creation of self
    :rtype: PCA
    """
    return self.pca

  def getFeaturesFrom(self, img):
    """
    return the singular values of shape n_components of the given matrix image
    by fiting a PCA with the given matrix image before

    :param img: a matrix list that represent an image
    :type img: list

    :return: ndarray of shape self.pca.n_components that contains the singular values of the given matrix image
    :rtype: ndarray
    """
    self.getPCA().fit(img)
    return self.getPCA().singular_values_