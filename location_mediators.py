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
    return distance

def generate_location_ideal():
    x=0
    y=0
    while x==0 and y==0:
        x= random.randint(0, 200)
        y= random.randint(0, 200)   
    return [x,y]


def generate_initial_location(agent_loc):
    mean = [agent_loc[0], agent_loc[1]] 
    sig1 = np.random.uniform(5, 40)
    sig2 = np.random.uniform(5, 40) 
    
    cov_matrix = [[sig1, 0], [0, sig2]]
    
    normal_dist = multivariate_normal(mean=mean, cov=cov_matrix)
    return abs(normal_dist.rvs(size=1))

def approve_proposal(sigma,proposal,ideal):
    approval=0
    if sigma>0:
        halfnorm_dist = halfnorm(scale=sigma)
        cdf_value = halfnorm_dist.cdf(l1_distance(proposal[0],ideal[0],proposal[1],ideal[1]))
        random_number = random.random()
        if random_number<=cdf_value:
            approval=1
    else:
        if l1_distance(proposal[0],ideal[0],proposal[1],ideal[1])<=l1_distance(0,ideal[0],0,ideal[1]):
            approval=1
    return approval


def create_agents(num_agents,sigma):
    agents=[]
    for i in range(num_agents):
        agent={}
        agent['agent_id']=i
        agent['ideal_location']=generate_location_ideal()
        agent['sigma']= sigma
        agents.append(agent)
    return agents
                                                         
def initialize_coalitions(agents):
    coalitions=[]
    for agent in agents:
        attribute={}
        attribute['proposal']= generate_initial_location(agent['ideal_location'])
        attribute['agents']=[]
        attribute['agents'].append(agent)
        coalitions.append(attribute)
    return coalitions


def mediator_func(two_coalitions):
    x=[]
    result_1 = [element * len(two_coalitions[0]['agents']) for element in two_coalitions[0]['proposal']]
    result_2 = [element * len(two_coalitions[1]['agents']) for element in two_coalitions[1]['proposal']]
    size=len(two_coalitions[0]['agents'])+len(two_coalitions[1]['agents'])
    return [(result_1[0]+result_2[0])/(size),(result_1[1]+result_2[1])/(size)]


def coalition_prop(coalitions, num_agents, alpha, booli):
    x_sum = sum(coalition['proposal'][0] * len(coalition['agents']) for coalition in coalitions)
    y_sum = sum(coalition['proposal'][1] * len(coalition['agents']) for coalition in coalitions)
    x_sum /= num_agents
    y_sum /= num_agents
    cols = []
    for i in range(len(coalitions)):
        dist = l1_distance(coalitions[i]['proposal'][0], x_sum, coalitions[i]['proposal'][1], y_sum)
        if booli:
            prob = 1 / (1 + alpha * dist)
        else:
            prob = 1 + alpha * dist
        cols.append({i: prob})
    total_prob_sum = sum(list(col.values())[0] for col in cols)
    normalized_probs = [{list(col.keys())[0]: list(col.values())[0] / total_prob_sum} for col in cols]
    event_labels = [list(col.keys())[0] for col in normalized_probs]
    event_probabilities = [list(col.values())[0] for col in normalized_probs]
    x = np.random.choice(event_labels, p=event_probabilities)
    new_coalitions = [coalition for i, coalition in enumerate(coalitions) if i != x]
    chosen_coalition = coalitions[x]
    distances = [l1_distance(chosen_coalition['proposal'][0], coalition['proposal'][0],
                             chosen_coalition['proposal'][1], coalition['proposal'][1]) for coalition in new_coalitions]
    min_distance = min(distances)
    closest_coalition = new_coalitions[distances.index(min_distance)]
    return [chosen_coalition, closest_coalition]

def coalition_formation(coalitions, dis, booli,alpha,num_agents):
    two_coalitions = coalition_prop(coalitions, num_agents, alpha, booli)
    proposal = mediator_func(two_coalitions)
    filtered_coalitions = []

    proposal_values = [tuple(coalition['proposal']) for coalition in two_coalitions]

    for coalition in coalitions:
        if tuple(coalition['proposal']) not in proposal_values:
            filtered_coalitions.append(coalition)

    attribute = {'proposal': proposal, 'agents': []}
    flag1 = []
    flag2 = []

    for agent in two_coalitions[0]['agents']:
        app = approve_proposal(sigma, proposal, agent['ideal_location'])
        if app == 1:
            flag1.append(agent)

    for agent in two_coalitions[1]['agents']:
        app = approve_proposal(sigma, proposal, agent['ideal_location'])
        if app == 1:
            flag2.append(agent)

    if not dis:
        attribute['agents'] = flag1 + flag2
    else:
        if (len(flag1) / len(two_coalitions[0]['agents'])) >= 0.5 and (len(flag2) / len(two_coalitions[1]['agents'])) >= 0.5:
            attribute['agents'] = flag1 + flag2

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

def simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents,booli, alpha,dis,sigma):
    itt = 0
    while not Halt(coalitions, num_agents)[0] and itt <10000:
        n_coalitions=coalition_formation(coalitions, dis, booli,alpha,num_agents)
        coalitions=n_coalitions
        itt += 1
        if itt >= 10000:
            times_av[key].append('no')
            q_dis[key].append('no')
            break
            
    if itt < 10000:
        times_av[key].append(itt)
        q_dis[key].append(calculate_avg_l1_distance(Halt(coalitions, num_agents)[1]))

def run_simulation(num_agents, sigma, times_av, q_dis,num):
    for booli in [False, True]:
        for alpha in [0,0.25,0.5,0.75,1]:
            for dis in [False, True]:
                key = (num_agents, booli, alpha,sigma,dis)
                if num==1:  
                    times_av[key] = []
                    q_dis[key]=[]
                agents = create_agents(num_agents,sigma)
                coalitions = initialize_coalitions(agents)
                simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents,booli, alpha,dis,sigma)

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
    for sigma in range (0,41,10):
        for num_agents in range(10, 41, 10):
            for num in range(1,51):
                run_simulation(num_agents, sigma, times_av, q_dis,num)
            
            
 
