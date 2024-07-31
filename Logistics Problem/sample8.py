# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:49:19 2024

@author: HMP_User
"""

# genetic algorithm build up

# how to encode
import numpy as np
from random import sample

def custom_repair_priority(path: list[tuple], priorityStations: list[tuple]) -> list[tuple]:
    if len(priorityStations) == 0:
        return path
    
    pathSet = set(path)
    for station in priorityStations:
        pathSet.discard(station)
        
    newPath = priorityStations[:] + list(pathSet)
    return newPath


def repair_genes(path: list[tuple], src: tuple) -> list[tuple]:
    path.remove(src)
    path.insert(0, src)
    
    return path


def population(size: int, requestingStations: list[tuple], src: tuple) -> list[list[tuple]]:
    startingPopulation = []
    
    for i in range(size):
        temp = list(map(tuple, np.random.permutation(requestingStations)))
        startingPopulation.append(repair_genes(temp, src))
    
    return startingPopulation


def merge_rules(rules):
    is_fully_merged = True
    for round1 in rules:
        if round1[0] == round1[1]:
            rules.remove(round1)
            is_fully_merged = False
        else:
            for round2 in rules:
                if round2[0] == round1[1]:
                    rules.append((round1[0], round2[1]))
                    rules.remove(round1)
                    rules.remove(round2)
                    is_fully_merged = False
    return rules, is_fully_merged



def crossover(path1: list[tuple], path2: list[tuple], src: tuple) -> list[tuple]:
    # src = path1[0]
    # path1 = path1[1:]
    # path2 = path2[1:]
    cxp1, cxp2 = sorted(sample(range(min(len(path1), len(path2))), 2))

    part1 = path2[cxp1: cxp2 + 1]
    part2 = path1[cxp1: cxp2 + 1]
    
    rule12 = list(zip(part1, part2))
    is_fully_merged = False
    
    while not is_fully_merged:
        rule12, is_fully_merged = merge_rules(rules = rule12)
        
    rule21 = {rule[1]: rule[0] for rule in rule12}
    rule12 = dict(rule12)
    
    ind1 = [gene if gene not in part2 else rule21[gene] for gene in path1[:cxp1]] + part2 + \
        [gene if gene not in part2 else rule21[gene] for gene in path1[cxp2 + 1:]]
    ind2 = [gene if gene not in part1 else rule12[gene] for gene in path2[:cxp1]] + part1 + \
        [gene if gene not in part1 else rule12[gene] for gene in path2[cxp2 + 1:]]
        
    ind1 = repair_genes(ind1, src)
    ind2 = repair_genes(ind2, src)
    return ind1, ind2


def mutate_index(offspring: list[tuple]) -> list[tuple]:
    if (size := len(offspring)) <= 2:
        return offspring
    
    start, stop = sorted(sample(range(1, size), 2))
    
    temp = offspring[start: stop + 1]
    temp.reverse()
    
    offspring[start: stop + 1] = temp
    return offspring


def evaluateFitness(path: list[tuple], distances: dict[dict[tuple]]) -> float:
    total_cost= 0
    
    for i in range(1, len(path)):
        total_cost += distances[path[i-1]][path[i]]
        
    fitness = 1/total_cost
    return fitness


def selection(poulation: list[list[tuple]], evaluation: list[int], parameter: int= 20) -> list[tuple]:
    betterChildren = sorted(evaluation, key= lambda x: x[1], reverse= True)
    
    # print(betterChildren)
    selectedChildren = [child for child, val in betterChildren]
    return selectedChildren[:parameter], betterChildren[:parameter]
    # return population[idx][:parameter], evaluation[idx][:parameter]


def generatPath(source: tuple, requestingStations: list[tuple], distances: dict[dict[tuple]], totalGenerations: int = 100, emergencyStations: list[tuple] = []) -> list[tuple]:
    requestingStations = list((source, )) + requestingStations
    # print(requestingStations)
    temp = population(len(requestingStations) + 1, requestingStations, source)
    
    offsprings = []
    best = []
    for i in range(totalGenerations):
        for i in range(len(temp)):
            i1, i2 = sorted(sample(range(len(temp)), 2))
            parent1, parent2 = temp[i1], temp[i2]
            
            child1, child2 = crossover(parent1, parent2, source)
            child1, child2 = mutate_index(child1), mutate_index(child2)
            
            child1 = custom_repair_priority(child1, emergencyStations)
            child2 = custom_repair_priority(child2, emergencyStations)
            
            offsprings.append((child1, evaluateFitness(child1, distances)))
            offsprings.append((child2, evaluateFitness(child2, distances)))
        temp, selectedPaths = selection(temp, offsprings)
        best.append(selectedPaths[0])
        
    return temp[0]
        