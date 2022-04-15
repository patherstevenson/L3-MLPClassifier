#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`DataLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: february 2022

DataLoader Module

"""
from FeatureLoader import FeatureLoader
import numpy as np
from tqdm import tqdm # progress bar

np.seterr(divide='ignore', invalid='ignore')

class DataLoader:
    """
    Create a DataLoader which will organize into different structures in order
    to improve their handling in function of what we want and 
    for which section (training or validation)
    """
    def __init__(self, dataimg,d_endindex):
        """
        Create a DataLoader which will organize into different structures in order
        to improve their handling in function of what we want and 
        for which section (training or validation)

        :param dataimg: dictionnary which must contains the matrix image data in train_img and valid_img key
        :param d_endindex: dictionnary which must contains the index for training slice and validation slice of data list image
        :type dataimg: dict 
        :type d_endindex: dict

        :UC: none
        """
        self.dataimg = dataimg
        self.d_endindex = d_endindex
        self.d_cat_index = {"train" : [], "valid" : []}
        self.d_img = {"train" : [], "valid" : []}
        self.d_features = {"train" : [], "valid" : []}
        self.sec = "train"
        self.fl = FeatureLoader()

    def initProgressBar(self):
        """
        :return: none
        :sideeffect: create a fresh tqdm progress bar bind to progressBar attrib
        """
        self.progressBar = tqdm()
        
    def incrementProgressBar(self):
        """
        :return: none:
        :sideeffect: increment by one the tqdm progress bar
        """
        self.progressBar.update(1)

    def setTotalProgressBar(self,total):
        """
        :param total: value to set as total max value of tqdm progress bar
        :type total: int

        :return: none
        :sideeffect: set attribute of tqdm progress bar as total given value
        """
        self.progressBar.total = total

    def getDataImg(self):
        """
        :return: dictionnary which contains the matrix image data
        :rtype: dict
        """
        return self.dataimg

    def getDictionnaryImg(self):
        """
        :return: dictionnary which contains the matrix image data organize by section
        :rtype: dict
        """
        return self.d_img

    def addInDictionnaryImg(self,section,img):
        """
        :return: none
        :sideeffect: append in the given section of the d_img dict the given image
        """
        self.getDictionnaryImg()[section].append(img)

    def getDictionnaryFeatures(self):
        """
        :return: dictionanry which contains matrix features extracted from matrix data image by using FeatureLoader
        :rtype: dict
        """
        return self.d_features

    def getFeatureLoader(self):
        """
        :return: the FeatureLoader object
        :rtype: FeatureLoader
        """
        return self.fl

    def addInDictionnaryFeatures(self,section,feature):
        """
        :return: none
        :sideeffect: append in the given section the given matrix feature of the d_features dict  
        """
        self.getDictionnaryFeatures()[section].append(feature)

    def getDictCategoriesIndex(self):
        """
        :return: dictionnary which contains couple (categorie, index) of image for each section train and valid
        :rtype: dict
        """
        return self.d_cat_index

    def addInDictCategorieIndex(self,section,cat,index):
        """
        :return: none
        :sideeffect: append in the given section of the d_cat_index dict the given tuple by (cat,index)
        """
        self.getDictCategoriesIndex()[section].append((cat,index))
    
    def getSection(self):
        """
        :return: the current section of self
        :rtype: str
        """
        return self.sec

    def setSection(self,section):
        """
        :return: none
        :sideeffect: set sec attribute to the given section if its equal to train otherwise valid
        """
        if section == "train":
            self.sec = section
        else:
            self.sec = "valid"

    def loadSectionFeaturesFromCategory(self,cat,section):
        """
        :param cat: the categorie
        :param section: the section
        :type cat: str
        :type section: str

        :return: none
        :sideeffect: fill d_img dict, d_cat_index and d_features by iterating on dataimg by append directly in and extract features with self.features (FeatureLoader object with PCA)

        :UC: section == 'train' or 'valid'
        """
        self.initProgressBar()
        self.setTotalProgressBar(self.d_endindex[cat][section])

        i = 0
        for img in self.getDataImg()[cat][section+"_img"]:
            self.addInDictionnaryImg(section,img)
            self.addInDictCategorieIndex(section,cat,i)

            f = self.getFeatureLoader().getFeaturesFrom(img)

            self.addInDictionnaryFeatures(section,f)
            self.incrementProgressBar()

            i += 1

    def convergeFeatures(self,categories,section):
        """
        :param categories: list of categories
        :param section: the section
        :type categories: list
        :type section: str

        :return: a first list which contains all features matrix and a second one with the categorie name of original image of each index of the first list
        :rtype: (list, list)
        """
        self.setSection(section)

        l_features = []
        l_cat = []

        for i in range(len(self.getDictCategoriesIndex()[self.getSection()])):
            (img, f, cat) = self[i]
            
            if cat not in categories:
                continue

            l_cat.append(cat)
            l_features.append(f)

        l_features = np.stack(l_features, axis=0)

        return l_features, l_cat

    def load(self):
        """
        For each categories this method will call loadSectionFeaturesFromCategory() method of self
        for train and valid section

        :return: none
        """
        print("\n\nDataLoader.load STARTED...",end='\n\n')

        for cat in self.getDataImg():

            print("\n\n\tEnter in category : " + cat,end='\n\n')
            print("LOAD FEATURES STARTING...")
            
            self.loadSectionFeaturesFromCategory(cat,"train")
            self.loadSectionFeaturesFromCategory(cat,"valid")

            print("DONE !")
        print("\n\nDataLoader.load DONE!",end='\n\n')

    def __getitem__(self,i):
        """
        :param i: the index of the wanted element in the d_list[self.sec] list
        :type i: int
        
        :return: a tuple that contains the following information about the image at the given i index (matrix, features, categorie)
        :rtype: tuple
        """
        cat, index = self.getDictCategoriesIndex()[self.getSection()][i]
        return self.getDictionnaryImg()[self.getSection()][i], self.getDictionnaryFeatures()[self.getSection()][i], cat