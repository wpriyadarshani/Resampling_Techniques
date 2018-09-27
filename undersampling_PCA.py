
from PIL import Image
import random
import numpy
import pdb

from PIL import Image

import array
import logging

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math


class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.pixels = []  # intialize pixels into a list
        self.centroid = None  # set the number of centro
        # ids to none

    def addPoint(self, pixel):  # add pixels to the pixel list
        self.pixels.append(pixel)


class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=3, max_PCA_iterations=20, min_distance=5.0, size=200, m=2.5, epsilon=.5, max_FCM_iterations=100):
        self.k = k  # initialize k clusters

        # intialize max_iterations
        self.max_PCA_iterations = max_PCA_iterations
        self.max_FCM_iterations = max_FCM_iterations

        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = size ** 2
        self.size = (size, size)  # intialize the size
        self.m = m
        self.epsilon = 0.01
        self.max_diff = 10.0
        self.image = 0
       
        self.pixels = []
        self.s_new = self.s 

 

    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels2 = numpy.array(image.getdata(), dtype=numpy.uint8)
        # self.beta = self.calculate_beta(self.image)

       

        # for i in range(self.s):
        #     self.pixels.append(self.pixels2[i])

        # for i in range(15):
        #     for i in range(len(self.SMOTE_array)):
        #         self.pixels.append(numpy.asarray(self.SMOTE_array[i]))


        # # set the size
        self.s_new = len(self.pixels)
          
        # print len(self.pixels), len(self.pixels2)
        # print "********** smote array size  ", len(self.SMOTE_array)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s_new):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))

        for i in range(self.s_new):
            num_1 = random.randint(1, 2) * 0.1
            num_2 = random.randint(1, 2) * 0.1
           
            num_3 = 1.0 - (num_1+num_2)
            degreelist = [num_1, num_2, num_3]
            self.degree_of_membership[i] = degreelist

        randomPixels = random.sample(self.pixels, self.k)
        print"INTIALIZE RANDOM PIXELS AS CENTROIDS"
        print randomPixels
        #    print"================================================================================"
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]
            # if(i ==0):
        for cluster in self.clusters:
            for pixel in self.pixels:
                cluster.addPoint(pixel)

        print "________", self.clusters[0].pixels[0]
        iterations = 0

        # FCM
        while self.shouldExitFCM(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membershipFCM()
           

            iterations += 1

        iterations = 0
        self.showClustering("FCM.png")
        # self.DB_index()

        # PCA
        while self.shouldExitPCA(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membershipPCA()
            
            iterations += 1

        for cluster in self.clusters:
            print cluster.centroid
        return [cluster.centroid for cluster in self.clusters]


    def selectSingleSolution(self):
        self.max_PCA_iterations = 10
        self.max_FCM_iterations=5


    def getClusterCentroid(self):
        centroid = []
        for cluster in self.clusters:
            centroid.append(cluster.centroid);

        return centroid

    def printClustorCentroid(self):
        for cluster in self.clusters:
            print cluster.centroid

    def shouldExitFCM(self, iterations):
        if self.max_diff<self.epsilon:
            return True

        if iterations <= self.max_FCM_iterations:
            return False
        return True

    def shouldExitPCA(self, iterations):
        if iterations <= self.max_PCA_iterations:
            return False
        return True

    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        for cluster in range(self.k):
            sum_numerator = 0.0
            sum_denominator = 0.0
            for i in range(self.s_new):
                pow_uij= pow(self.degree_of_membership[i][cluster], self.m)
                sum_denominator +=pow_uij
                num= pow_uij * self.pixels[i]

                sum_numerator+=num

            updatedcluster_center = sum_numerator/sum_denominator

            self.clusters[cluster].centroid = updatedcluster_center


    # Updates the degree of membership for all of the data points.
    def update_degree_of_membershipFCM(self):
        self.max_diff = 0.0

        for idx in range(self.k):
            for i in range(self.s_new):
                new_uij = self.get_new_value(self.pixels[i], self.clusters[idx].centroid)
                if (i == 0):
                    print "This is the Updatedegree centroid number:", idx, self.clusters[idx].centroid
                diff = new_uij - self.degree_of_membership[i][idx]
                if (diff > self.max_diff):
                    self.max_diff = diff
                self.degree_of_membership[i][idx] = new_uij
        return self.max_diff

    def get_new_value(self, i, j):
        sum = 0.0
        val = 0.0
        p = (2 * (1.0) / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in self.clusters:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / denom
            val = pow(val, p)
            sum += val
        return (1.0 / sum)

    def getEta(self, idx):
        sum_membership = 0.0
        eta_numerator = 0.0
        eta_k = 1.0

        for i in range(self.s_new):
            dis = pow(self.calcDistance(self.clusters[idx].centroid, self.pixels[i]), 2.0)

            membership_power = pow(self.degree_of_membership[i][idx], self.m)

            eta_numerator += (membership_power * dis)

            sum_membership += membership_power

        eta =eta_numerator / sum_membership
        eta = eta * eta_k
        return eta

    # update the degree of membership for PCA
    def update_degree_of_membershipPCA(self):
        #PCA 96
        for idx in range(self.k):
            eta = 0.0
            eta_k = 1.0

            #get eta for particular cluster
            eta = self.getEta(idx)

            if eta > 0.0:
                # print "******************* eta", eta
                for i in range(self.s_new):
                    if (i == 0):
                        print "This is the Update degree centroid number:", idx, self.clusters[idx].centroid

                    dis = pow(self.calcDistance(self.clusters[idx].centroid, self.pixels[i]), 2.0)

                    factor = dis / eta
                    factor = factor * -1.0

                    updated_membership_degree = math.exp(factor)

                    self.degree_of_membership[i][idx] = updated_membership_degree

    def undersampling(self):
        image_GT = Image.open('GT_T4.png')
        pixels_GT = numpy.array(image_GT.getdata(), dtype=numpy.uint8)


        #open second image
        image_org = Image.open('T4.png')
        pixels_org = numpy.array(image_org.getdata(), dtype=numpy.uint8)


        GT_pixelmap = image_GT.load()
        Org_pixelmap  = image_org.load()

        factor = 0.2
      #get white pixel
        j1 = (200 * 15) + 157

        #red
        m = 0

        #green value

        r = 39999

        
        pixels_mountain = []
        pixels_river = []
        pixels_village = []

        for i in range(200):
            for j in range(200):
                if self.calcDistance(GT_pixelmap[i,j], pixels_GT[j1]) == 0.0: 

                    pixel = [int(Org_pixelmap[i, j][0]) , int(Org_pixelmap[i, j][1] ), int(Org_pixelmap[i, j][2]), 255]
                    pixels_village.append(pixel)

        #mountain
        for i in range(200):
            for j in range(200):
                if self.calcDistance(GT_pixelmap[i,j], pixels_GT[m]) == 0.0: 
                    pixel = [int(Org_pixelmap[i, j][0]) , int(Org_pixelmap[i, j][1] ), int(Org_pixelmap[i, j][2]), 255]
                    pixels_mountain.append(pixel)

        #river
        for i in range(200):
            for j in range(200):
                if self.calcDistance(GT_pixelmap[i,j], pixels_GT[r]) == 0.0: 
                    pixel = [int(Org_pixelmap[i, j][0]) , int(Org_pixelmap[i, j][1] ), int(Org_pixelmap[i, j][2]), 255]
                    pixels_river.append(pixel)

               
        print "length of oversample " , len(pixels_village), len(pixels_mountain), len(pixels_river) 

        rand_mountain = []
        rand_river = []

        for i in range(900):
            r = random.randint(0,25333)
            rand_river.append(pixels_river[r])
            # rand_mountain.append(pixels_mountain[r])

        for i in range(900):
            r = random.randint(0,13765)
            rand_mountain.append(pixels_mountain[r])

        print "length of oversample " , len(rand_mountain), len(rand_river) 


        for i in range(900):
            self.pixels.append(numpy.asarray(rand_mountain[i]))
            self.pixels.append(numpy.asarray(rand_river[i]))
        
        for j in range(1):     
            for i in pixels_village:
                self.pixels.append(numpy.asarray(i))


    def showClustering(self, name):
        localPixels = [None] * len(self.image.getdata())
        for idx, pixel in enumerate(self.pixels2):
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster

            # if nearest == self.clusters[0]:
            #     localPixels[idx]=[229,75,77]
            # elif nearest == self.clusters[1]:
            #     localPixels[idx] = [56,129,78]
            # elif nearest == self.clusters[2]:
            #     localPixels[idx] = [251,227,227]
            localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels) \
            .astype('uint8') \
            .reshape((h, w, 4))
        colourMap = Image.fromarray(localPixels)
        # colourMap.show()

        plt.imsave(name, colourMap)




if __name__ == "__main__":
    image = Image.open("T4.png")
    f = fcm()

    f.undersampling()
    result = f.run(image)


    f.showClustering("PCA.png")

    # f.normalizemembership()
    # # print f.I_index()
    # # # print f.JmFunction()
    # # print f.XBindex()

    # f.normalization()
    # print f.DB_index()
