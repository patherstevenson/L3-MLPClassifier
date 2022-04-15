#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`DataLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: march 2022

GenerateTest Module

"""

import numpy as np
import random
import matplotlib.pyplot as plt 

class GenerateTest:
  """
  Create a GenerateTest which can generate a set of passed and failed tests for each data categories
  """
  def __init__(self, dataloader,hpt,cl):
    """
    Create a GenerateTest which can generate a set of passed and failed tests for each data categories
    for the given DataLoader, Hyperparametertuning and Categoriesloader used
  
    :param dataloader: the used Dataloader
    :param hpt: the used HyperparamterTuning
    :param cl: the used CategoriesLoader
    :type dataloader: DataLoader
    :type hpt: HyperparameterTuning
    :type cl: CategoriesLoader
    """
    self.dl = dataloader
    self.hpt = hpt
    self.cl = cl

    self.initDict()

  def initDict(self):
    """
    initialize the needed dictionnaries for test generation

    :return: none
    """
    self.d_test = {k : {cat: [] for cat in self.cl.getCategories()} for k in ['found','fail']}
    self.d_label = {k : {cat: [] for cat in self.cl.getCategories()} for k in ['found','f_sup','f_expec']}
    self.d_check = {k : [False for cat in self.cl.getCategories()] for k in ['found', 'fail']}
    self.d_cat_index = {self.cl.getCategories()[i] : i for i in range(len(self.cl.getCategories()))}

  def getHpt(self):
    """
    :return: the HyperparamterTuning object given at the creation of self
    :rtype: HyperparameterTuning
    """
    return self.hpt

  def verifySupposedCategorie(self,img,supposed,cat):
    """
    verify if the mlp found the right categorie for the given image and fill dictionnaries
    in function of if it's fail or not. If a fail or a success as be already done for the given
    categorie then do nothing

    :param img: matrix image
    :param supposed: supposed categorie of the given img by the mlp
    :param cat: the categorie of the given img stored at during the preprocessing
    :type img: ndarray
    :type supposed: str
    :type cat: str

    :return: none
    """
    res = (supposed == cat)
    if ((not res) and (not self.d_check['fail'][self.d_cat_index[cat]])):
      if len(self.d_test['fail'][cat]) > 0:
        self.d_check['fail'][self.d_cat_index[cat]] = True
      self.d_test['fail'][cat].append(img)
      self.d_label['f_expec'][cat].append(cat)
      self.d_label['f_sup'][cat].append(supposed)
    
    elif not self.d_check['found'][self.d_cat_index[cat]]:
      if len(self.d_test['found'][cat]) > 0:
        self.d_check['found'][self.d_cat_index[cat]] = True
      self.d_test['found'][cat].append(img)
      self.d_label['found'][cat].append(cat)

  def generateImageTest(self):
    """
    Generate test image in directory data/res/found and data/res/fail by using matplotlib.pyplot

    For found tests this show the image and the supposed categorie of the mlp for the image
    
    Otherwise for the fail tests of mlp this show image, the supposed categorie of the mlp
    and the categorie of the image saved during the preprocessing which is the right one.

    :return: none
    """
    print("GenerateTest.generateImageTest STARTING...",end='\n\n')
    found,fail = 0,0

    for cat in self.cl.getCategories():
      try:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(self.d_test['found'][cat][0])
        ax.titlesize = 80
        ax.set_title('MLP Found :\n {}'.format(self.d_label['found'][cat][0]))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('data/res/found/found_{}.png'.format(found))
        plt.close()
        found += 1
      except:
        print('except found -> {}'.format(cat))
        continue
      try:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(self.d_test['fail'][cat][0])
        ax.titlesize = 80
        ax.set_title('MLP Fail\nSupposed : {}\nExpected : {}'.format(self.d_label['f_sup'][cat][0],self.d_label['f_expec'][cat][0]))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('data/res/fail/fail_{}.png'.format(fail))
        plt.close()
        fail += 1
      except:
        print('except fail -> {}'.format(cat))
        continue

    print("GenerateTest.generateImageTest -> folder : data/res/found & data/res/fail",end='\n\n')

  def startTest(self,select):
    """
    Start a test session for the slice of mlp in the sub list hpt[select:]
    in range of total number of data categories.

    Until we found a found and fail examples for each data categories of this slice of mlp
    we iterate on dataloader to get the tuple (img,features,cat) and we use the lda of the current
    mlp used in hpt[select:] list to transform the features and use them in the predict method of mlp.

    Then we give the predict method return to the call of verifySupposedCategorie() with img and cat.

    Finally the call the generateImageTest() method which generate the image of the results of the tests

    :param select: index of the first mlp to use in the HyperparameterTuning MLP list
    :type select: int

    :return: none
    """
    print("GenerateTest.startTest STARTING...",end='\n\n')

    self.dl.setSection("valid")

    l_mlp = self.getHpt().getMLPList()[select:]
    
    for h in range(self.getHpt().max_layer // self.getHpt().min_layer):
      if (np.array(self.d_check['found']).all() and np.array(self.d_check['fail']).all()):
        break
      else:
        mlp, lda = l_mlp[h]
        len_dl = len(self.dl.getDictCategoriesIndex()[self.dl.getSection()])

        for i in range(len_dl):
          (img, f, cat) = self.dl[random.randint(0,len_dl-1)]
          features = lda.transform(f.reshape(1,32))
          supposed = mlp.predict(features.reshape(-1,1))

          self.verifySupposedCategorie(img,supposed[0],cat)
    
    print("\t TEST DONE !",end='\n\n')

    self.generateImageTest()

    print("GenerateTest.startTest DONE !",end='\n\n')