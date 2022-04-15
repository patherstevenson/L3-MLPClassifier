#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`MatrixLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: janvier 2022

MatrixLoader Module

"""

from imageio import imread
import os

class MatrixLoader:
    """
    Create a clean MatrixLoader associated for a given path folder and categories list
    """
    def __init__(self,path,categories):
        """
        :param path: path of the folder where to find the training image by categories
        :type path: str
        :param categories: a list that contains the categories of the path folder
        :type categories: str

        :build: a clean MatrixLoader associated with the given path folder and categories list

        :UC: type(path) == str & type(categories) == list
        """
        assert(type(path) == str)
        assert(type(categories) == list)

        self.path = path
        self.categories = categories
        self.d_img = {cat : [] for cat in categories}
        self.validation_p = 0.75
        self.d_endindex = {cat : {'train' : 0, 'valid' : 0} for cat in categories}
        self.data = {}

    def getDictionnaryImg(self):
        """
        :return: the dictionnary which contains couple (category,img_name)
        :rtype: dict
        """
        return self.d_img

    def getDataImg(self):
        """
        :return: the dictionnary which contains couple (category,dict) where the dict will contains some list of valid image, tr
        :rtype: dict
        """
        return self.data

    def getDictionnaryEndIndex(self):
        """
        :return: the dictionnary which contains the index for training slice and validation slice in the data list
        :rtype: dict
        """
        return self.d_endindex

    def addInDictionnaryImg(self,categorie,img_id):
        """
        :return: None
        :side effect: add the given img_id path to d_img dict with the given categorie as key
        """
        self.d_img[categorie].append(os.path.join(self.path+categorie,img_id))

    def fillDictionnaryImg(self):
        """
        :return: None
        :side effect: adding all founded img in the given path for each categories
        """
        print("fillDictionnaryImg STARTING...",end='\n\n')

        for cat in self.categories:
            if self.path[-1] != "/":
                self.path += "/"
            for img in os.listdir(self.path+cat):
                self.addInDictionnaryImg(cat,img)

        print("fillDictionnaryImg DONE!",end='\n\n')

    def generateTrainAndValidMatrixImg(self):
        """
        create set of training and valid img

        imread create a matrix of greyscale of given img

        :return: None
        :side effect: fill the data dict by category in order to eliminate a part of training img to use them in the validation phase of the model
        """
        self.fillDictionnaryImg()

        print("generateTrainAndValidMatrixImg STARTING...",end='\n\n')
        print("PORTION OF VALIDATION SET = {:.2f}%".format((1-self.validation_p)*100),end='\n\n')

        for cat in self.categories:
            print("Enter in category : "+cat,end='\n\n')

            l_img = self.d_img[cat]

            self.getDictionnaryEndIndex()[cat]['train'] = int(len(l_img) * self.validation_p)
            self.getDictionnaryEndIndex()[cat]['valid'] =  len(l_img) - self.d_endindex[cat]['train']

            print("\ttraining_end index = " + str(self.d_endindex[cat]['train']),end='\n\n')
            print("\tvalid_end index = " + str(len(l_img)),end='\n\n')

            training_file = l_img[:self.getDictionnaryEndIndex()[cat]['train']]
            valid_file = l_img[self.getDictionnaryEndIndex()[cat]['train']:]

            training_img = [imread(img) for img in training_file] # create a matrix for all training image path given in training_file list 
            valid_img = [imread(img) for img in valid_file] #create a matrix for all validation image path given in valid_file list 

            self.data[cat] = {'train_file': training_file,
                              'valid_file': valid_file,
                              'train_img' : training_img,
                              'valid_img' : valid_img}

            print("DATA DICTIONNARY FILLED FOR CATEGORY -> "+ cat,end='\n\n')

        print("generateTrainAndValidMatrixImg DONE!", end='\n\n')
