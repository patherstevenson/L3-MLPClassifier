#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`DataLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: march 2022

HyperparameterTuning Module

"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import TrainingModel
import random

class HyperparameterTuning:
    
    """
    Create a HyperparamaterTuning which will create different trained mlp with differents subset of data categories 
    in order to select overall the best set of mlp for a given number of categories
    """
    def __init__(self, dataLoader, categories):
        """
        Create a HyperparamaterTuning which will create different trained mlp with differents subset of data categories 
        in order to used the overall best set of mlp for a given number of categories

        :param dataloader: the DataLoader used during the preprocessing dataset loading
        :param categories: the list of data categories
        :type dataloader: DataLoader
        :type categories: List
        """
        self.dl = dataLoader
        self.categories = categories
        self.len_cat = len(categories)
        self.l_mlp = []
        self.train_accuracy = []
        self.valid_accuracy = []
        self.random_accuracy = [1/n for n in range(2,self.len_cat + 1)]
        self.min_layer = 8
        self.max_layer = 32

    def getMLPList(self):
        """
        :return: the list of all MLPClassifier used in the grid search session
        :rtype: list
        """
        return self.l_mlp

    def getRandomAccuracy(self,i):
        """
        :return: the random accuracy at the given index i of the random_accuracy list
        :rtype: float
        """
        return self.random_accuracy[i]

    def startIterate(self):
        """
        This method will start iterate on [2,3,..,n] where n = total number of data categories
        and on 8 to 32 by a step of 8 that represent the number of hidden layer of the trained 
        mlp for the current state of the double loop n,hidden_layer. 
        
        Then we train this mlp for a random choice of n categories in the list of all 
        data categories except if n is at his last iteration in which case we give all categories
        to the mlp.

        We print and save the accuracy for train and valid section and random accuracy in function of 
        the number of categories.

        :return: none
        """

        print("HyperparameterTuning.start_iterate STARTED...",end='\n\n')
        print("\tTRAIN & VALIDATION ACCURACY RUN:\n")

        for i in range(2,self.len_cat + 1):
            print("\n\t\tNB_CAT {} :".format(i),end='\n\n')

            for hidden_layer in range(self.min_layer, (self.max_layer + self.min_layer), self.min_layer):
                res = (0,0)
                
                sub_cat = self.categories if (i == self.len_cat) else np.random.choice(self.categories, i, replace=False)

                mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(hidden_layer, hidden_layer), random_state=1,max_iter=10000, warm_start=True, early_stopping=True)

                t, v, lda = TrainingModel.train(mlp, self.dl, sub_cat)

                res = (t, v)
                self.getMLPList().append((mlp, lda))

                self.train_accuracy.append(res[0])
                self.valid_accuracy.append(res[1])

                print("\t\t\t\ttrain {} valid {} rand {} -> {}".format(res[0], res[1], self.getRandomAccuracy(i-2),sub_cat),end='\n')
                
        print("\nHyperparameterTuning.start_iterate DONE",end='\n\n')

