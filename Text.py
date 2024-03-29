import random
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
os.environ["TFHUB_CACHE_DIR"] = r"C:\Users\User\Downloads\New folder"
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
from openai import OpenAI
import string
import re
def outputTreat(text):
    sentences = re.split(r'\. *\n|\.\n', text)  # Split based on '. \n' or '.\n'
    sentence_dict = {}
    for i, sentence in enumerate(sentences, 1):
        cleaned_sentence = re.sub(r'^\d+\) ?', '', sentence).strip('. ').strip()
        if cleaned_sentence:  # Exclude empty sentences
            sentence_dict[i] = cleaned_sentence
    return {key: value for key, value in sentence_dict.items()}

def LLM(prompt,system_message):
    client = OpenAI(
        api_key="##################################"
    )

    open_ai_payload = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.75,
    }

    completion = client.chat.completions.create(
        stream=False, **open_ai_payload
    )

    response_data = completion.choices[0].message.content
    return response_data.strip()

def embed(input):
    return model(input)


def combined_dist(vector1, vector2):
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return math.sqrt(2-2*similarity)



def approve_proposal(sigma, proposal, ideal, status_quo):
    approval = 0
    if combined_dist(proposal, ideal) < combined_dist(status_quo, ideal):
        approval = 1
    elif combined_dist(proposal, ideal) >= combined_dist(status_quo, ideal) and sigma > 0:
        lower_bound = 0  
        upper_bound = 2 
        a = (lower_bound - 1) / sigma
        b = (upper_bound - 1) / sigma
        truncated_dist = truncnorm(a, b, loc=1, scale=sigma)
        pdf_value = truncated_dist.pdf(combined_dist(proposal, ideal))
        random_number = random.uniform(0,1)
        if random_number <= pdf_value:
            approval = 1
    return approval


def create_agents(num,sigma):
    agents=[]
    message= f'You are an assistant, give a straight froward answer with no introduction,  give your answers with 1) 2) 3) and so on for each sentence you propose '
    promt=f'Give me {num} different sentences that are well structured about how to deal whith global warming each with at most of 15 words.'
    sen=list(outputTreat(LLM(promt,message)).values())
    sentence_embeddings=embed(sen)
    for i in range(num):
        agent={}
        agent['agent_id']=i
        agent['ideal_location']=sentence_embeddings[i]
        agent['ideal_sentence']=sen[i]
        agent['sigma']= sigma
        agents.append(agent)
    return agents
                                                         
def initialize_coalitions(agents):
    coalitions = []
    for agent in agents:
        attribute = {}
        attribute['proposal'] = agent['ideal_location'] 
        attribute['sentence'] = agent['ideal_sentence']
        attribute['agents'] = []
        attribute['agents'].append(agent)
        coalitions.append(attribute)
    return coalitions


def mediator_func(two_coalitions):
    promt=f'Give me 5 possible different well structured sentences, that aggregate following two sentences : 1) {two_coalitions[0]["sentence"]} 2) {two_coalitions[1]["sentence"]}. Make sure each sentence  has at most 15 words.'
    message= f'You are a mediator trying to find agreed wording for how to deal with global warming based on existing sentences. Give a straight froward answer with no introduction in-order to help people reach an agreed wording of a coherent sentence. number your answers (i.e 1),2),3),4),5) and so on) for each sentence you propose.'
    g=LLM(promt,message)
    sentences = list(outputTreat(g).values())
    embedded = embed(sentences)
    centroid = (len(two_coalitions[0]['agents']) * two_coalitions[0]['proposal'] + len(two_coalitions[1]['agents']) * two_coalitions[1]['proposal']) / (len(two_coalitions[0]['agents']) + len(two_coalitions[1]['agents']))
    min_distance = float('inf')  
    closest_vector = None
    closest_index = -1

    for idx, vector in enumerate(embedded):
        dist = combined_dist(centroid, vector)
        
        if dist < min_distance:
            min_distance = dist
            closest_vector = vector
            closest_index = idx
    return [embedded[closest_index], sentences[closest_index]]                                                                                                                                                 
                                                                                                                                                     
                                                                                                                                           
                                                                                                                                        
                                                                                                                                                    
def coalition_prop(coalitions, num_agents, alpha, booli):
    centroid = sum(coalition['proposal'] * len(coalition['agents']) for coalition in coalitions)
    centroid/=num_agents
    cols = []
    for i in range(len(coalitions)):
        dist = combined_dist(centroid, coalitions[i]['proposal'])
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
    distances = [combined_dist(chosen_coalition['proposal'], coalition['proposal']) for coalition in new_coalitions]
    min_distance = min(distances)
    closest_coalition = new_coalitions[distances.index(min_distance)]
    return [chosen_coalition, closest_coalition]

def coalition_formation(coalitions, dis, booli,alpha,num_agents,status_quo):
    two_coalitions = coalition_prop(coalitions, num_agents, alpha, booli)
    proposal = mediator_func(two_coalitions)
    filtered_coalitions = []
    proposal_values = [tuple(coalition['proposal']) for coalition in two_coalitions]

    for coalition in coalitions:
        if not any(np.array_equal(coalition['agents'], c['agents']) for c in two_coalitions):
            filtered_coalitions.append(coalition)

    attribute = {'proposal': proposal[0],'sentence': proposal[1], 'agents': []}
    flag1 = []
    flag2 = []

    for agent in two_coalitions[0]['agents']:
        app = approve_proposal(sigma, proposal[0], agent['ideal_location'],status_quo)
        if app == 1:
            flag1.append(agent)

    for agent in two_coalitions[1]['agents']:
        app = approve_proposal(sigma, proposal[0], agent['ideal_location'],status_quo)
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
        'sentence':two_coalitions[0]['sentence'],
        'agents': [a for a in two_coalitions[0]['agents'] if a not in flag1]
    }
    t2 = {
        'proposal': two_coalitions[1]['proposal'],
        'sentence':two_coalitions[1]['sentence'],
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

def simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents, booli, alpha, dis, sigma, status_quo):
    itt = 0
    while not Halt(coalitions, num_agents)[0] and itt < 1000:
        n_coalitions = coalition_formation(coalitions, dis, booli, alpha, num_agents, status_quo)
        coalitions = n_coalitions
        itt += 1
        if itt >= 1000:
            times_av[key].append('no')
            q_dis[key].append('no')
            break

    if itt < 1000:
        times_av[key].append(itt)
        ppp = Halt(coalitions, num_agents)[1]
        q_dis[key].append(calculate_avg_l1_distance(ppp))

def run_simulation(num_agents, sigma, times_av, q_dis, num):
    for booli in [False, True]:
        for alpha in [0, 0.33, 0.67, 1]:
            for dis in [False, True]:
                key = (num_agents, booli, alpha, sigma, dis)
                if key not in list(t.keys()):
                    sen = ['There is no sentence since the coalition formation has not started yet',
                            'There is no sentence since the coalition formation has not started yet']
                    status_quo = embed(sen)[0]
                    if num == 1:
                        times_av[key] = []
                        q_dis[key] = []

                    agents = create_agents(num_agents, sigma)
                    coalitions = initialize_coalitions(agents)
                    while True:
                        try:
                            simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents, booli, alpha, dis,sigma, status_quo)
                            break  # Break the loop if successful
                        except Exception as e:
                            print(f"Retry due to error: {e}")

def run_with_retry(num_agents, sigma, times_av, q_dis, num):
    while True:
        try:
            run_simulation(num_agents, sigma, times_av, q_dis, num)
            break  # Break the loop if successful
        except Exception as e:
            print(f"Retry due to error: {e}")

def calculate_avg_l1_distance(coalition):
    proposal = coalition['proposal']
    distances = []
    for agent in coalition['agents']:
        agent_location = agent['ideal_location']
        distance = combined_dist(proposal,agent_location)
        distances.append(distance)
    avg_distance = sum(distances) / len(distances)
    return avg_distance


            
if __name__ == "__main__":
    times_av = {}
    q_dis = {}
    for sigma in [0,1,1.5,2]:
        for num_agents in [10,20,30,40]:
            for num in range(1,11):
                run_with_retry(num_agents, sigma, times_av, q_dis, num)
