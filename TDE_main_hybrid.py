"""


"""

import random
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
import Automold as am
import os
import cv2 as cv2
import utils_self

population_size = 5
generations = 100
IMG_DIM = 608
F = 0.5
CR = 0.6

lower_bound = np.array([0., 0.05, 0., 0., 3, 0., 0., 0., 0.])
upper_bound = np.array([0.3, 0.3, 1., 1., 8, 1., 1., 1., 1.])


variable_dimension = len(lower_bound)
OBJ_CONF = 0.4  


cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "chectpoint"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()


def init_population():

    population = np.zeros((population_size, variable_dimension))

    for i in range(population_size):

            

        for j in range(variable_dimension):

            if (j == 4):

                v = np.random.randint(int(lower_bound[j]), int(upper_bound[j]))
            elif (j==7 or j==8):
                rand_value = np.random.rand()
                v = population[i, j-2] + rand_value * (upper_bound[j] - population[i, j-2])
            else:
                rand_value = np.random.rand()
                v = lower_bound[j] + rand_value * (upper_bound[j] - lower_bound[j])
            population[i, j] = v


    return population


def calculate_fitness(target_image, population):

    fitness = []
    model.eval()
    imgshape = target_image.shape


    for b in range(population_size):


        population_iter = population[b]

        img_automolded = target_image
        img_automolded = am.add_snow(target_image, snow_coeff=population_iter[0])

        img_automolded = am.add_fog(img_automolded, fog_coeff=population_iter[1])

        img_automolded = am.add_sun_flare(img_automolded, flare_center=(
            population_iter[2]*imgshape[1], population_iter[3]*imgshape[0]))

        rect = (int(population_iter[5]*imgshape[1]),int(population_iter[6]*imgshape[0]), int(population_iter[7]*imgshape[1]),int(population_iter[8]*imgshape[0]))
        img_automolded = am.add_shadow(img_automolded, rectangular_roi=rect, shadow_dimension=int(population_iter[4]))


        attack_image = utils_self.img_transfer(img_automolded)
        attack_image.clamp_(0, 1)
        attack_image = transforms.ToPILImage('RGB')(attack_image.cpu())

        outputs_boxes = utils.do_detect(
            model, attack_image, 0.4, 0.4, True)

        f_score = 0.0

        if len(outputs_boxes) == 0:
            fitness.append([-1])
        else:
            outputs_obj_conf = torch.Tensor(outputs_boxes)
            all_obj_conf = outputs_obj_conf[:, 4]
            f_score = all_obj_conf - OBJ_CONF
            fitness.append(f_score)
 

    return fitness


def mutation(population):
    Mpopulation = np.zeros((population_size, variable_dimension))

    for i in range(population_size):
        r1 = r2 = r3 = 0
        F_temp = random.random()  
        if F_temp > 0.5:
            F = 2
        else:
            F = 0.5

        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])

        for j in range(variable_dimension):
            
            if (j == 4):
                if lower_bound[j] <= Mpopulation[i, j] <= upper_bound[j]:
                    Mpopulation[i, j] = int(Mpopulation[i, j])
                else:
                    Mpopulation[i, j] = np.random.randint(int(lower_bound[j]), int(upper_bound[j]))   
                    
            elif (j==7 or j==8):
                if Mpopulation[i, j-2]< Mpopulation[i, j]:
                    if lower_bound[j] < Mpopulation[i, j] < upper_bound[j]:
                        Mpopulation[i, j] = Mpopulation[i, j]
                    else:
                        rand_value = np.random.rand()
                        Mpopulation[i, j] = population[i, j-2] + rand_value * (upper_bound[j] - population[i, j-2])
                else:
                    rand_value = np.random.rand()
                    Mpopulation[i, j] = population[i, j-2] + rand_value * (upper_bound[j] - population[i, j-2])
                               
            elif lower_bound[j] <= Mpopulation[i, j] <= upper_bound[j]:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i, j] = lower_bound[j] + \
                    random.random() * (upper_bound[j] - lower_bound[j])
            

    return Mpopulation


def crossover(Mpopulation, population):

    Cpopulation = np.zeros((population_size, variable_dimension))
    for i in range(population_size):
        for j in range(variable_dimension):
            rand_float = random.random()
            
            if rand_float <= CR:
                
                Cpopulation[i, j] = Mpopulation[i, j]
            else:
                Cpopulation[i, j] = population[i, j]
            if (j==7 or j==8):
                if Cpopulation[i, j-2] > Cpopulation[i, j]:
                    rand_value = np.random.rand()
                    Cpopulation[i, j] = Cpopulation[i, j-2] + rand_value * (upper_bound[j] - Cpopulation[i, j-2])


    return Cpopulation


def selection(taget_image, Cpopulation, population, pfitness):

    Cfitness = calculate_fitness(taget_image, Cpopulation)  
    for i in range(population_size):
        if len(Cfitness[i]) <= len(pfitness[i]):  
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness


def fitness_selection(fitness):

    fitness_len = []
    for items in fitness:
        if (len(items) == 1 and items[0] == -1):
            fitness_max_value = -1
            fitness_len.append(len(items))
        else:
            fitness_len.append(len(items))
    fitness_min_len = min(fitness_len)  

    selected_index = [i for i, x in enumerate(
        fitness_len) if x == fitness_min_len]  
    
    select_list = []  
    for i in range(len(selected_index)):
        select_list.append(max(fitness[selected_index[i]]))
    fitness_index = selected_index[np.argmin(select_list)]  
    fitness_index_value = fitness[fitness_index]  
    fitness_max_value = max(fitness_index_value)  
    return fitness_index, fitness_max_value


def FDE(clean_image):

    imgshape = clean_image.shape

    population = init_population()   
  
    fitness = calculate_fitness(clean_image, population)  
    
    fitness_index, fitness_max_value = fitness_selection(fitness)
    Best_indi = population[fitness_index]

    for step in range(generations):
        if fitness_max_value < 0:
            print("break step : ", step)
            break
        
        Mpopulation = mutation(population)
       
        Cpopulation = crossover(Mpopulation, population)
    
        print("step : ", step, "max fitness : ", fitness_max_value)

        population, fitness = selection(
            clean_image, Cpopulation, population, fitness)
        fitness_index, fitness_max_value = fitness_selection(fitness)
        
        Best_indi = population[fitness_index]

    img_automolded = am.add_snow(clean_image, snow_coeff=Best_indi[0])
    img_automolded = am.add_fog(img_automolded, fog_coeff=Best_indi[1])

    img_automolded = am.add_sun_flare(img_automolded, flare_center=(
        Best_indi[2]*imgshape[1], Best_indi[3]*imgshape[0]))
    
    rect = (int(Best_indi[5]*imgshape[1]),int(Best_indi[6]*imgshape[0]), int(Best_indi[7]*imgshape[1]),int(Best_indi[8]*imgshape[0]))
    img_automolded = am.add_shadow(img_automolded,  rectangular_roi=rect, shadow_dimension=int(Best_indi[4]))

    final_image = utils_self.img_transfer(img_automolded)
    
    final_image.clamp_(0, 1)

    return final_image


