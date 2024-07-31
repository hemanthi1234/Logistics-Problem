# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:59:57 2024

@author: HMP_User
"""

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from sklearn.cluster import KMeans
import sample8
# from numba import jit

startTime = perf_counter()
# V = int(input("Enter the size of the graph: "))
V = 15
plt.figure(figsize=(V*2/3, V/2))
graph = defaultdict(lambda : defaultdict(lambda: math.inf))

obstacles = [tuple(np.random.randint(0,V, size = 2)) for _ in range(np.random.randint(V,5*V))]
if (3,4) in obstacles:
    obstacles.remove((3,4))
    
for x1 in range(V):
    for y1 in range(V):
        for x2 in (-1,0,1):
            for y2 in (-1,0,1):
                if (x1, y1) == (x1 + x2, y1 + y2):
                    continue
                if (x1, y1) in obstacles or (x1 + x2, y1 + y2) in obstacles:
                    graph[(x1, y1)][(x1 + x2, y1 + y2)] = math.inf
                    graph[(x1 + x2, y1 + y2)][(x1, y1)] = math.inf
                    continue
                graph[(x1,y1)][(x1 + x2, y1 + y2)] = graph[(x1 + x2, y1 + y2)][(x1, y1)] = 1.414 if abs(x2) == abs(y2) else 1
                


requests = []

for x in range(V):
    if x != V-1:
        grid, = plt.plot([i for i in range(0, V)], [x + 0.5] * V, c= 'black', scalex= True, scaley= True)
    for y in range(V):
        if y != V-1:
            plt.plot([y+0.5] * V, [i for i in range(0, V)], c= 'black', scalex= True, scaley= True)
        if (x,y) == (3,4):
            plt.scatter(x, y, c="green", marker= '*', label= 'hub')
            continue
        if (x,y) in obstacles:
            continue
        if np.random.choice([True, False], p=[0.2, 0.8]):
            if np.random.choice([True, False]):
                stations = plt.scatter(x, y, c= "red")
                requests.append((x,y))
            else:
                others = plt.scatter(x, y, c= "blue")
        else:
            plt.scatter(x, y, c= "white")
            
# plt.plot([i+0.5 for i in range(0, V)])
# plt.show()
# plt.legend()



plt.scatter(np.array(obstacles)[:,0], np.array(obstacles)[:,1], color = "black", marker = "s", label= 'obstacles')
# print(graph)


def floydWarshall():
    dist = defaultdict(lambda: defaultdict(int))
    pred = defaultdict(lambda: defaultdict(int))
    for x1 in range(V):
        for y1 in range(V):
            for x2 in range(V) :
                for y2 in range(V):
                    i = (x1, y1)
                    j = (x2, y2)
                    dist[i][j] = graph[i][j]
                    pred[i][j] = j
                        
    # print(dist[(0,0)][(3,4)])
        
    for x3 in range(V):
        for y3 in range(V):
            for x1 in range(V):
                for y1 in range(V):
                    for x2 in range(V):
                        for y2 in range(V):
                            k = (x3, y3)
                            i = (x1, y1)
                            j = (x2, y2)
                            if dist[i][j] > dist[i][k] + dist[k][j]:
                                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                                pred[i][j] = pred[i][k]
                
    # print("Distance matrix: ", dist)
    # print(pred)
    return dist, pred


def generateRequest(graph, source):
    request = defaultdict(lambda: ["None", 0])
    
    for x in range(V):
        for y in range(V):
            station = (x, y)
            if station == source:
                continue
            request[station] = [np.random.choice(["Medicines", "Food", "Fuel", "Ammunations", "None"], p= [0.1, 0.3, 0.1, 0.2, 0.3]), np.random.randint(100,10000)]
            if request[station][0] == "None":
                request[station][1] = 0
    # print(request)
    return request



def combinedRequest(dayCount, source):
    # import json
    
    request = defaultdict(lambda: ["None", 0] * V)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in days[:dayCount]:
        request[day] = generateRequest(graph, source)
        
    # print(request)
    for req in request:
        # imp = False
        # print(req)
        for needs in request[req]:
            if request[req][needs][0] == "None":
                continue
            # print("\t\t Outpost", needs, "needs", request[req][needs][0])
            if request[req][needs][0] == "Medicines":
                pass
                # imp = True
                # print("Important, sending support right away\n")
        # imp = False
    return request
        


def getRoute(graph, source, req):
    path = defaultdict(int)
    distances, predecessors = floydWarshall()
    # # this will not work, we need all sources shortest path
    
    # # assuming we have that
    
    path = [source]
    
    remaining = req
    
    currentStation = source
    
    '''
        # this is not going to work, as this time we have to travel more
        # than just the active nodes, so greedy is not an option
        # therefore, for finding the path, we will have to explore the 
        # entire path, and then, perhaps use dynamic programming to find
        # out the least cost path
        # this is really trouble some... 
        # perhaps bfs or dfs can be used to traverse through the graph
        # and then maybe we can use dp to figure out the best path
        # I am not at all sure whether this will even work or not.
    '''
    # print(remaining)
    while remaining:
        nextStation = req[0]
        minDistance = float('inf')
        for x in range(V):
            for y in range(V):
                station = (x, y)
                distance = distances[currentStation][station]
                # print(distance)
                if station not in remaining:
                    continue
                if distance < minDistance:
                    minDistance= distance
                    nextStation= station
        path.append(nextStation)
        # print(nextStation)
        remaining.remove(nextStation)
        currentStation = nextStation
        
    # we still need edge weights to update and create the paths accordingly
    # therefore, it is best to leave this for now, because edge weights is actually a dependency
    # take the intial path, and then from the active stations, 
    # we can distirbute them into something like replenished and remanining sets
    # and then we can pick the minimum from remaining and go the next
    return path, predecessors
    
    
# @jit
if __name__ == "__main__":
    # for i in range(V):
    #     graph[i] = list(set(list(np.random.randint(0,V,np.random.randint(0,V)))))
    
    # points = []
    # print(graph)
    # V = V\
    requestingStations = requests
    points = requests
    # requestingStations = points[:]
    src = (3, 4)
    # req = generateRequest(graph, src)
    combinedRequest(5, src)
    
    
    # what if reaching to a station is not possible, that is the given path is
    # completely blocked for some reason, do we need to consider for that condition?
    # if yes, then sending out multiple vehicles, adaptability, flexibility and 
    # the dynamic nature should come into picture
    # '''
    # n = int(input("Enter the number of vehicles: "))
    n = 3
    model = KMeans(n_clusters= n, n_init= "auto")
    model.fit(points)
    seperatedPoints = [[] for _ in range(n)]
    
    labels = model.labels_
    for pi, label in enumerate(labels):
        seperatedPoints[label].append(requestingStations[pi])
    
    tg = int(input("Enter the number of generations: "))
    
    distances, pred = floydWarshall()
    for j in range(n):
        # seperatedPath, sepPred = getRoute(graph, src, seperatedPoints[j])
        seperatedPath, sepPred = sample8.generatPath(src, seperatedPoints[j][:], distances, totalGenerations= tg), pred
        # print("\n The path to be taken is: ")
        # for station in seperatedPath:
            # print(station, end = " -> ")
        
        expandedPath = []
        for i in range(1, len(seperatedPath)):
            u = seperatedPath[i-1]
            v = seperatedPath[i]
            expandedPath.append(u)
            while u != v:
                u = sepPred[u][v]
                expandedPath.append(u)
        
        # print(expandedPath)
        plt.plot(np.array(expandedPath)[:, 0], np.array(expandedPath)[:, 1], label = f'path for vehicle {j + 1}')
    
    # '''
    
    # path, predecessors = getRoute(graph, src, points)
    # print("\n The path to be taken is: ")
    # for station in path:
    #     print(station, end = " -> ")
    
    # expandedPath = []
    # for i in range(1, len(path)):
    #     u = path[i-1]
    #     v = path[i]
    #     expandedPath.append(u)
    #     while u != v:
    #         u = predecessors[u][v]
    #         expandedPath.append(u)
    
    # # print(expandedPath)
    # plt.plot(np.array(expandedPath)[:, 0], np.array(expandedPath)[:, 1], label = 'path')
    
    # pathPoints = []
    # for station in path:
    #         pathPoints.append(station)
    # plt.plot(np.array(pathPoints)[:,0], np.array(pathPoints)[:,1])

    
    # plt.scatter(points[0][0], points[0][1], marker= '*', color= 'green')
    # # print(req)
    # for i in range(V):
    #     if i == 0:
    #         continue
    #     if req[i][0] != "None":
    #         plt.scatter(x=points[i][0], y= points[i][1], color= 'yellow', marker= '^')
    #     if req[i][0] == "Medicines":
    #         plt.scatter(x=points[i][0], y= points[i][1], color= 'red', marker= 'o')
    

    # plt.plot(list(range(20)), [-2] * 20)
    # plt.plot([-2] * 20, list(range(20)))
    stations.set_label('active requests')    
    grid.set_label('gridlines')
    others.set_label('stations')
    plt.legend(loc= 'upper right')
    plt.xlabel("x-coordinates")
    plt.ylabel("y-coordinates")
    # plt.Figure(figsize= (200, 200))
    plt.show()
    
    dis = floydWarshall()[0]
    # print(points)
    ga_path = sample8.generatPath(src, requestingStations[:], dis)
    gr_path = getRoute(graph, src, requestingStations[:])[0]
    print("\nGenerationally preferred path: ", ga_path, "with Fitness value of: ", sample8.evaluateFitness(ga_path, dis))
    print("\nGreedy Approach path:", gr_path, "with Fitness value: ", sample8.evaluateFitness(gr_path, dis))
    endTime = perf_counter()
    print(f"Time of Execution= {endTime - startTime}")


# execute()
