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
import math
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
        api_key=###########
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



def create_agents(num, sigma):
    agents = []
    message = 'You are an assistant, give a straightforward answer with no introduction, give your answers with 1), 2), 3), and so on for each sentence you propose.'
    prompt = f'Give me {num} different sentences that are well-structured about how to deal with global warming, each with at most 15 words.'
    sentences = list(outputTreat(LLM(prompt, message)).values())
    
    # Embed all sentences at once for efficiency
    sentence_embeddings = embed(sentences)
    
    for i in range(num):
        agent = {}
        agent['agent_id'] = i
        agent['ideal_location'] = sentence_embeddings[i]  # Store the embedding here
        agent['ideal_sentence'] = sentences[i]  # Store the actual sentence text here
        agent['sigma'] = sigma
        agents.append(agent)
    
    return agents

def approve_proposal(sigma, proposal, ideal, status_quo):
    approval = 0
    if combined_dist(proposal, ideal) < combined_dist(status_quo, ideal):
        approval = 1
    elif combined_dist(proposal, ideal) >= combined_dist(status_quo, ideal) and sigma > 0:
        lower_bound = 0
        upper_bound = 2
        truncated_dist = truncnorm(lower_bound, upper_bound, loc=0, scale=sigma)
        pdf_value = truncated_dist.pdf(combined_dist(proposal, ideal))
        random_number = random.uniform(0, 1)
        if random_number <= pdf_value:
            approval = 1
    return approval

                                                         
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


def mediator_func(two_coalitions,base):
    if base==0:
        message= f'You are an assistant, give a straight froward answer with no introduction,  give your answers with 1) 2) 3) and so on for each sentence you propose '
        promt=f'Give me 2 different sentences that are well structured about random things with at most of 15 words.'
    elif base==1:
        promt=f'Give me 2 possible different well structured sentences, that aggregate following two sentences : 1) {two_coalitions[0]["sentence"]} 2) {two_coalitions[1]["sentence"]}. Make sure each sentence  has at most 15 words.'
        message= f'You are a mediator trying to find agreed wording for how to deal with global warming based on existing sentences. Give a straight forward answer with no introduction in-order to help people reach an agreed wording of a coherent sentence. number your answers (i.e 1),2),3),4),5) and so on) for each sentence you propose.'
    elif base==2:
        promt=f'Give me 10 possible different well structured sentences, that aggregate following two sentences : 1) {two_coalitions[0]["sentence"]} 2) {two_coalitions[1]["sentence"]}. Make sure each sentence  has at most 15 words.'
        message=f'You are a mediator trying to find agreed wording for how to deal with global warming based on existing sentences. Give a straight forward answer with no introduction in-order to help people reach an agreed wording of a coherent sentence. number your answers (i.e 1),2),3),4),5) and so on) for each sentence you propose.'
    elif base==3:
        promt= f'Generate 10 concise and clear sentences that blend the following two sentences into one coherent idea: 1) {two_coalitions[0]["sentence"]} 2) {two_coalitions[1]["sentence"]}. Ensure each sentence is no longer than 15 words. number your answers (i.e 1),2),3),4),5) and so on) for each sentence you propose'
        message= f'As a mediator, you need to find a consensus on global warming solutions. Provide straightforward and numbered suggestions to help reach a clear and agreed-upon sentence.'
    else:
        promt= f'Create 10 unique, well-structured sentences that combine these two sentences into one unified thought: 1) {two_coalitions[0]["sentence"]} 2) {two_coalitions[1]["sentence"]}. Each sentence should be a maximum of 15 words. number your answers (i.e 1),2),3),4),5) and so on) for each sentence you propose'
        message=  f'You are acting as a mediator to achieve a common statement on global warming. Give direct and numbered suggestions to assist in forming a unified and coherent sentence.'
    g=LLM(promt,message)
    sentences = list(outputTreat(g).values())
    embedded = embed(sentences)
    if base==0 or base==1:
        return [embedded[0],sentences[0]]
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
                                                                                                                                                     
                                                                                                                                                                                                                                                            
def combined_dist(vector1, vector2):
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    similarity = np.clip(similarity, -1, 1) 
    return math.sqrt(2 - 2 * similarity)

def coalition_prop(coalitions, num_agents, booli):
    centroid = sum(np.array(coalition['proposal']) * len(coalition['agents']) for coalition in coalitions) / num_agents
    cols = []
    for i in range(len(coalitions)):
        dist = combined_dist(centroid, coalitions[i]['proposal'])
        cols.append({i: dist})
    max_distance = max(list(col.values())[0] for col in cols)
    normalized_distances = [{list(col.keys())[0]: list(col.values())[0] / max_distance} for col in cols]
    scores = [np.exp(booli * (list(col.values())[0])) for col in normalized_distances]
    total_score_sum = sum(scores)
    normalized_probs = [score / total_score_sum for score in scores]
    chosen_index = np.random.choice(len(coalitions), p=normalized_probs)
    chosen_coalition = coalitions[chosen_index]
    new_coalitions = [coalition for i, coalition in enumerate(coalitions) if i != chosen_index]
    distances = [combined_dist(chosen_coalition['proposal'], coalition['proposal']) for coalition in new_coalitions]
    min_distance = min(distances)
    closest_coalition = new_coalitions[distances.index(min_distance)]

    return [chosen_coalition, closest_coalition]



def coalition_formation(coalitions, dis, booli,num_agents,status_quo,base):
    two_coalitions = coalition_prop(coalitions, num_agents, booli)
    proposal = mediator_func(two_coalitions,base)
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


def simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents, booli, dis, sigma, status_quo,base):
    itt = 0
    coalition_sentences = []
    while not Halt(coalitions, num_agents)[0] and itt < 1000:
        n_coalitions = coalition_formation(coalitions, dis, booli, num_agents, status_quo, base)
        coalitions = n_coalitions
        coalition_sentences.append([{agent['agent_id']: agent['ideal_sentence']} for coalition in coalitions for agent in coalition['agents']])
        itt += 1
        if itt >= 1000:
            times_av[key].append('no')
            q_dis[key].append('no')
            break

    if itt < 1000:
        times_av[key].append(itt)
        ppp = Halt(coalitions, num_agents)[1]
        q_dis[key].append(calculate_avg_l1_distance(ppp))
        return coalitions, coalition_sentences


def run_simulation(num_agents, sigma, times_av, q_dis, num,base):
    coalition_sentences_all_iterations = []  
    for booli in [0,1]:
        for dis in [True,False]:
            key = (num_agents, booli, sigma, dis)
            sen = ['There is no sentence since the coalition formation has not started yet',
                    'There is no sentence since the coalition formation has not started yet']
            status_quo = embed(sen)[0]
            if num == 0:
                times_av[key] = []
                q_dis[key] = []

            agents = create_agents(num_agents, sigma)
            coalitions = initialize_coalitions(agents)
            coalitions, coalition_sentences = simulate_coalition_formation(times_av, q_dis, coalitions, key, num_agents, booli, dis, sigma, status_quo,base)
            coalition_sentences_all_iterations.append(coalition_sentences)
    return agents, coalitions, coalition_sentences_all_iterations


def calculate_avg_l1_distance(coalition):
    proposal = coalition['proposal']
    distances = []
    for agent in coalition['agents']:
        agent_location = agent['ideal_location']
        distance = combined_dist(proposal,agent_location)
        distances.append(distance)
    avg_distance = sum(distances) / len(distances)
    return avg_distance


t={}
q={}
for base in [0,1,2,3,4]:
    times_av = {}
    q_dis = {}
    for num in range(20):
        for sigma in [0,1]:
            agents, coalitions,coalition_sentences_all_iterations=run_simulation(10,sigma, times_av, q_dis, num,base)
    t[base]=times_av.values()
    q[base]=q_dis.values

from scipy import stats

means = {key: np.mean(generated_data[key], axis=0) for key in generated_data}

mean_values = [means[key] for key in sorted(means.keys())]
keys = list(sorted(means.keys()))

f_stat, p_val = stats.f_oneway(*mean_values)

print(f"One-way ANOVA results:")
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")
if p_val < 0.05:
    print("ANOVA indicates significant differences between groups.")
    tukey_results = stats.tukey_hsd(np.concatenate(mean_values), np.concatenate([[i] * len(mean_values[i]) for i in range(len(mean_values))]))
    print("\nTukey's HSD post-hoc test results:")
    print(tukey_results)
    rankings = np.argsort([np.mean(means[key]) for key in keys])

    print("\nRanking based on Tukey's HSD test:")
    for rank, idx in enumerate(rankings):
        print(f"{rank + 1}. Key {keys[idx]}: Mean = {np.mean(means[keys[idx]]):.4f}")
else:
    print("ANOVA does not indicate significant differences between groups.")

