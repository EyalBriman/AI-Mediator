import random
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
def l1_distance(x1,x2,y1,y2):
    return abs(x2 - x1) + abs(y2 - y1)

def generate_location_init(cor_x,cor_y):
    x=0
    y=0
    while l1_distance(cor_x,x,cor_y,y)>=l1_distance(cor_x,0,cor_y,0): 
        x= random.randint(0, 200)
        y= random.randint(0, 200) 
    return [x,y]

def generate_location_ideal():
    x=0
    y=0
    while x==0 and y==0:
        x= random.randint(0, 200)
        y= random.randint(0, 200)   
    return [x,y]



def approve_proposal(sigma,proposal,ideal):
    approval=0
    normal_dist = norm(loc=0, scale=sigma)
    cdf_value = normal_dist.cdf(l1_distance(proposal[0],ideal[0],proposal[1],ideal[1]))
    random_number = random.random()
    if random_number<=cdf_value:
        approval=1
    return approval


def propose_new_coalitions_random(coalitions):
    return random.sample(coalitions,2)

def propose_new_coalitions_minmax(coalitions):
    min_coalition = min(coalitions, key=lambda x: len(x['agents']))
    max_coalition = max(coalitions, key=lambda x: len(x['agents']))
    return [min_coalition, max_coalition]

def propose_new_coalitions_centorid(coalitions,num_agents):
    x_sum=0
    y_sum=0
    for coalition in coalitions:
        result = [element * len(coalition['agents']) for element in coalition['proposal']]
        x_sum+=result[0]
        y_sum+=result[1]
    x_sum=x_sum/num_agents
    y_sum=y_sum/num_agents
    closest_proposals = {'closest1': None, 'closest2': None}
    closest_distances = [float('inf'), float('inf')]
    for coalition in coalitions:
        proposal = coalition['proposal']
        distance = l1_distance(x_sum, proposal[0], y_sum, proposal[1])
        if distance < closest_distances[0]:
            closest_distances[1] = closest_distances[0]
            closest_proposals['closest2'] = closest_proposals['closest1']
            closest_distances[0] = distance
            closest_proposals['closest1'] = proposal
        elif distance < closest_distances[1]:
            closest_distances[1] = distance
            closest_proposals['closest2'] = proposal
    return closest_proposals['closest1'], closest_proposals['closest2']
        
                                                                                              
def create_agents(num_agents):
    agents=[]
    for i in range(num_agents):
        agent={}
        agent['agent_id']=i
        agent['ideal_location']=generate_location_ideal()
        agent['sigma']= random.uniform(6, 10)
        agents.append(agent)
    return agents
                                                         
def initialize_coalitions(agents):
    coalitions=[]
    for agent in agents:
        attribute={}
        attribute['proposal']=generate_location_init(agent['ideal_location'][0],agent['ideal_location'][1])
        attribute['agents']=[]
        attribute['agents'].append(agent)
        coalitions.append(attribute)
    return coalitions

def mediator_func(two_coalitions,agg):
    x=[]
    if agg:
        result_1 = [element * len(two_coalitions[0]['agents']) for element in two_coalitions[0]['proposal']]
        result_2 = [element * len(two_coalitions[1]['agents']) for element in two_coalitions[1]['proposal']]
        size=len(two_coalitions[0]['agents'])+len(two_coalitions[1]['agents'])
        x=[(result_1[0]+result_2[0])/(size),(result_1[1]+result_2[1])/(size)]
    else:
        x=[(two_coalitions[0]['proposal'][0]+two_coalitions[1]['proposal'][0])/(2),(two_coalitions[0]['proposal'][1]+two_coalitions[1]['proposal'][1])/(2)]
    return x
  
                                                         
def coalition_formation(coalitions, rnd, weighted_agg, dis, num_agents):
    if rnd == 1:
        two_coalitions = propose_new_coalitions_random(coalitions)
    elif rnd == 2:
        two_coalitions = propose_new_coalitions_minmax(coalitions)
    else:
        two_coalitions = propose_new_coalitions_centorid(coalitions, num_agents)
    proposal = mediator_func(two_coalitions, weighted_agg)
    attribute = {'proposal': proposal, 'agents': []}

    if dis > 0:
        coalition1_approve = sum(approve_proposal(agent['sigma'], proposal, agent['ideal_location'])
                                 for agent in two_coalitions[0]['agents'])
        coalition2_approve = sum(approve_proposal(agent['sigma'], proposal, agent['ideal_location'])
                                 for agent in two_coalitions[1]['agents'])

        if coalition1_approve / len(two_coalitions[0]['agents']) >= dis and \
                coalition2_approve / len(two_coalitions[1]['agents']) >= dis:
            coalitions = [coalition for coalition in coalitions if coalition not in two_coalitions[:2]]
            attribute['agents'].extend(two_coalitions[0]['agents'])
            attribute['agents'].extend(two_coalitions[1]['agents'])
            coalitions.append(attribute)
    else:
        for agent in two_coalitions[0]['agents']:
            if approve_proposal(agent['sigma'], proposal, agent['ideal_location']) == 1:
                attribute['agents'].append(agent)
        for agent in two_coalitions[1]['agents']:
            if approve_proposal(agent['sigma'], proposal, agent['ideal_location']) == 1:
                attribute['agents'].append(agent)

        coalitions = [coalition for coalition in coalitions if coalition not in two_coalitions[:2]]
        coalitions.append(attribute)

    return [coalition for coalition in coalitions if len(coalition['agents']) >= 1]

def Halt(coalitions,num_agents):
    flag = False
    for coalition in coalitions:
        if (len(coalition['agents'])/num_agents)>0.5:
            flag=True
    return flag

def visualize_coalitions_per_iteration(coalitions_history):
    colors = ['blue', 'red', 'green', 'orange']  # You can extend this list for more colors if needed

    for step, coalitions in enumerate(coalitions_history):
        plt.figure(figsize=(8, 8))
        plt.title(f'Coalitions at Step {step}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        for coalition in coalitions:
            agent_ids = [agent['agent_id'] for agent in coalition['agents']]
            x, y = coalition['proposal']
            plt.scatter(x, y, color=colors[step % len(colors)], label=f'Coalition {agent_ids}')

        plt.xlim(0, 200)  # Adjust the limits based on your coordinate system
        plt.ylim(0, 200)  # Adjust the limits based on your coordinate system
        plt.legend(title='Agent IDs in Coalitions', loc='upper left')
        plt.grid(True)
        plt.show()

# ... (existing code)

if __name__ == "__main__":
    num_agents = 4
    rnd = 1
    dis = 0
    weighted_agg = True
    agents = create_agents(num_agents)
    coalitions = initialize_coalitions(agents)
    coalitions_history = [coalitions.copy()]

    while not Halt(coalitions, num_agents):
        coalitions = coalition_formation(coalitions, rnd, weighted_agg, dis, num_agents)
        coalitions_history.append(coalitions.copy())
    print(coalitions)
    visualize_coalitions_per_iteration(coalitions_history)
