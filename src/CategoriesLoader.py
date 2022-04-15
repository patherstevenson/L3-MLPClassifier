#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`CategoriesLoader` module
:author: Pather Stevenson - Faculté des Sciences et Technologies - Univ. Lille <http://portail.fil.univ-lille1.fr>_
:date: janvier 2022

CategoriesLoader Module

"""

class CategoriesLoader:
    """
    Create a CategoriesLoader which will read the data/descriptions/categories.txt in order to
    get the list of data categories 

    >>> cl = CategoriesLoader("../data/descriptions/categories.txt")
    >>> cl.getNbCategories()
    0
    >>> cl.getCategories()
    []
    >>> cl.readCategories()
    >>> cl.getNbCategories()
    7
    >>> cl.getCategories()
    ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    >>> cl.showCategories()
    angry
    disgust
    fear
    happy
    neutral
    sad
    surprise
    """

    def __init__(self, path):
        """
        Create a CategoriesLoader which will read the data/descriptions/categories.txt in order to
        get the list of data categories 

        :param path: path of file where analyse categories
        :type path: string
        :build: a clean CategoriesLoader associated to the given file path

        :UC: type(path) == str
        """
        assert(type(path) == str)

        self.nb_categories = 0
        self.path = path
        self.categories = []

    def getCategories(self):
        """
        :return: the list of categories of this CL
        :rtype: list
        """
        return self.categories

    def getNbCategories(self):
        """
        :return: the number of categories of this CL
        :rtype: int
        """
        return self.nb_categories

    def showCategories(self):
        """
        :return: None
        :side effect: print the categories list with line breaks
        """
        print("showCategories : ",end="\n\n\t- ")
        print(*self.categories, sep="\n\t- ",end="\n\n")

    def foundCategories(self):
        """
        :return: None
        :side effect: read the given file path and add categories name to the categories list of this CL
        :UC: self.path must exist
        :raise: :class: `FileNotFoundError` if self.path is a nonexistent file
        """
        print("foundCategories STARTING...",end='\n\n')

        with open(self.path,'r') as f:
            self.categories = f.read().splitlines()
        self.nb_categories = len(self.categories)

        print("foundCategories DONE!",end='\n\n')
        print("NB founded categories = "+ str(self.nb_categories),end='\n\n')
        
        self.showCategories()