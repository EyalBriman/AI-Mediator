import random
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
from scipy.stats import multivariate_normal
import time

def l1_distance(point1x, point2x,point1y, point2y):
    distance = abs(point1x - point2x) + abs(point1y - point2y)
    return (distance)

def generate_location_ideal(mus,sigmas,weights):
    chosen_component1 = np.random.choice(peaks, p=weights)
    chosen_component2 = np.random.choice(peaks, p=weights)
    sample_x = np.random.normal(mus[chosen_component1], sigmas[chosen_component1])
    sample_y = np.random.normal(mus[chosen_component2], sigmas[chosen_component2])

    return [sample_x, sample_y]


def generate_initial_location(agent_loc):
    mean = [agent_loc[0], agent_loc[1]] 
    sig1 = np.random.uniform(5, 40)
    sig2 = np.random.uniform(5, 40) 
    
    cov_matrix = [[sig1, 0], [0, sig2]]
    
    normal_dist = multivariate_normal(mean=mean, cov=cov_matrix)
    return abs(normal_dist.rvs(size=1))

def approve_proposal(sigma,proposal,ideal,rx,ry):
    approval=0
    if l1_distance(proposal[0],ideal[0],proposal[1],ideal[1])<=l1_distance(rx,ideal[0],ry,ideal[1]):
        approval=1
    elif  l1_distance(proposal[0],ideal[0],proposal[1],ideal[1])>=l1_distance(rx,ideal[0],ry,ideal[1]) and sigma>0:
        halfnorm_dist = halfnorm(scale=sigma)
        pdf_value = halfnorm_dist.pdf(l1_distance(proposal[0],ideal[0],proposal[1],ideal[1]))
        random_number = random.uniform(0, 1)
        if random_number<=pdf_value:
            approval=1
    return approval


def create_agents(num_agents,sigma,peaks):
    agents=[]
    mus = np.random.uniform(0, 200, peaks)
    sigmas = np.random.uniform(1, 50, peaks)
    weights = np.random.dirichlet(np.ones(peaks))
    for i in range(num_agents):
        agent={}
        agent['agent_id']=i
        if peaks==0:
            agent['ideal_location']=[random.uniform(0, 200),random.uniform(0, 200)]
        else:
            agent['ideal_location']=generate_location_ideal(mus,sigmas,weights)
        agent['sigma']= sigma
        agents.append(agent)
    return agents
                                                         
def initialize_coalitions(agents,noise):
    coalitions = []
    for i, agent in enumerate(agents):
        attribute = {}
        if noise:
            attribute['proposal'] = agent['ideal_location']
        else:
            attribute['proposal']=generate_initial_location(agent['ideal_location'])
        attribute['agents'] = []
        attribute['agents'].append(agent)
        coalitions.append(attribute)
    return coalitions


def mediator_func(two_coalitions):
    x=[]
    result_1 = [element * len(two_coalitions[0]['agents']) for element in two_coalitions[0]['proposal']]
    result_2 = [element * len(two_coalitions[1]['agents']) for element in two_coalitions[1]['proposal']]
    size=len(two_coalitions[0]['agents'])+len(two_coalitions[1]['agents'])
    return [(result_1[0]+result_2[0])/(size),(result_1[1]+result_2[1])/(size)]


def coalition_prop(coalitions, num_agents, booli):
    x_sum = sum(len(coalition['agents']) * coalition['proposal'][0] for coalition in coalitions)
    y_sum = sum(len(coalition['agents']) * coalition['proposal'][1] for coalition in coalitions)
    x_sum /= num_agents
    y_sum /= num_agents
    centroid = (x_sum, y_sum)
    distances = [l1_distance(coalition['proposal'][0], centroid[0], coalition['proposal'][1], centroid[1]) for coalition in coalitions]
    max_distance = max(distances)
    normalized_distances = [dist / max_distance for dist in distances]
    scores = [np.exp(booli * (dist)) for dist in normalized_distances]
    total_score_sum = sum(scores)
    probabilities = [score / total_score_sum for score in scores]
    chosen_index = np.random.choice(len(coalitions), p=probabilities)
    chosen_coalition = coalitions[chosen_index]
    distances.pop(chosen_index)
    min_distance = min(distances)
    closest_coalition = coalitions[distances.index(min_distance)]
    return chosen_coalition, closest_coalition

def coalition_formation(coalitions, dis, booli,num_agents,rx,ry):
    two_coalitions = coalition_prop(coalitions, num_agents, booli)
    proposal = mediator_func(two_coalitions)
    filtered_coalitions = []

    for coalition in coalitions:
        if not any(np.array_equal(coalition['agents'], c['agents']) for c in two_coalitions):
            filtered_coalitions.append(coalition)

    attribute = {'proposal': proposal, 'agents': []}
    flag1 = []
    flag2 = []

    for agent in two_coalitions[0]['agents']:
        app = approve_proposal(sigma, proposal, agent['ideal_location'],rx,ry)
        if app == 1:
            flag1.append(agent)

    for agent in two_coalitions[1]['agents']:
        app = approve_proposal(sigma, proposal, agent['ideal_location'],rx,ry)
        if app == 1:
            flag2.append(agent)

    if not dis:
        attribute['agents'] = flag1 + flag2
    else:
        if (len(flag1) / len(two_coalitions[0]['agents'])) >= 0.5 and (len(flag2) / len(two_coalitions[1]['agents'])) >= 0.5:
            attribute['agents'] = flag1 + flag2
        else:
            return coalitions

    t1 = {
            'proposal': two_coalitions[0]['proposal'],
            'agents': [a for a in two_coalitions[0]['agents'] if a not in flag1]
        }
    t2 = {
            'proposal': two_coalitions[1]['proposal'],
            'agents': [a for a in two_coalitions[1]['agents'] if a not in flag2]
        }

    if attribute['agents']:
        filtered_coalitions.append(attribute)
    if t1['agents']:
        filtered_coalitions.append(t1)
    if t2['agents']:
        filtered_coalitions.append(t2)

    return [coalition for coalition in filtered_coalitions if len(coalition['agents']) > 0]


def Halt(coalitions,num_agents):
    coalitionBig=coalitions[0]
    flag = False
    for coalition in coalitions:
        if len(coalition['agents'])/num_agents>=0.5 or len(coalitions)==2:
            flag=True
            coalitionBig=coalition
    return [flag,coalitionBig]

def simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents,booli,dis,sigma,rx,ry):
    itt = 0
    while not Halt(coalitions, num_agents)[0] and itt <5000:
        n_coalitions=coalition_formation(coalitions, dis, booli,num_agents,rx,ry)
        coalitions=n_coalitions
        itt += 1
        if itt >= 5000:
            times_av[key].append('no')
            q_dis[key].append('no')
            break
            
    if itt < 5000:
        times_av[key].append(itt)
        q_dis[key].append(calculate_avg_l1_distance(Halt(coalitions, num_agents)[1]))

def run_simulation(num_agents, sigma, times_av, q_dis,num,rx,ry,peakes):
    for t in [-1,0,1]:
        for C in [False, True]:
            for I in [True,False]:
                n=num_agents
                key = (n,t,peaks,sigma,C,I)
                if num==1:  
                    times_av[key] = []
                    q_dis[key]=[]
                agents = create_agents(n,sigma,peaks)
                coalitions = initialize_coalitions(agents,I)
                simulate_coalition_formation(times_av, q_dis, coalitions, key, n,t,C,sigma,rx,ry)

def calculate_avg_l1_distance(coalition):
    proposal = coalition['proposal']
    distances = []
    for agent in coalition['agents']:
        agent_location = agent['ideal_location']
        distance = l1_distance(proposal[0], agent_location[0], proposal[1], agent_location[1])
        distances.append(distance)
    avg_distance = sum(distances) / len(distances)
    return avg_distance


            
if __name__ == "__main__":
    times_av = {}
    q_dis = {}
    for peaks in [0,1,2,3,4]:
        for sigma in [0,10,20,30]:
            for num_agents in range(10, 41, 10):
                for num in range(1,101):
                    rx= random.uniform(0, 200)
                    ry= random.uniform(0, 200)
                    run_simulation(num_agents, sigma, times_av, q_dis,num,rx,ry,peaks)

            
 
