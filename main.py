from transformers import pipeline
import random
import itertools
import numpy as np
import spacy
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
nlp = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer"])
def embed_sentence(sentence, nlp,stop_words):
    doc = nlp(sentence)
    word_embeddings = []
    for token in doc:
        if not stop_words or token.text.lower() not in stop_words:
            if token.has_vector:
                word_embeddings.append(token.vector)
    if word_embeddings:
        return sum(word_embeddings) / len(word_embeddings)
    else:
        return None
def generate_sentences_agent(num_agents):#need to build
    sentences=[]
    return sentences

def generate_sentences_init(num_agents):#need to build
    sentences=[]
    return sentences

# Function to simulate an agent's approval/disapproval decision using the transformer
def approve_proposal(agent,proposal):
    approve=false
    # each agent approves or disapproves of the proposal 
    return approve

# Functions to propose 2 new coalitions
def propose_new_coalitions_random(coalitions):
    return random.sample(coalitions,2)

def propose_new_coalitions_minmax(coalitions):
    min_coalition = min(coalitions, key=lambda x: len(x['agents']))
    max_coalition = max(coalitions, key=lambda x: len(x['agents']))
    return [min_coalition, max_coalition]
                                        
                                                         
def create_agents(num_agents):#need to build
    agents=[]
    sentences=generate_sentences_agent(num_agents)#also this
    for i in range(len(sentences)):
        attribute={}
        attribute['agent_id']=i
        attribute['ideal_sentence']=sentences[i]
        attribute['chat_instance']=GPT()#this is the thing to build
    return agents
                                                         
def initialize_coalitions(agents):
    coalitions=[]
    sentences=generate_sentences_init(len(agents))
    for i in range(len(sentences)):
        attribute={}
        attribute['proposal']=sentence
        attribute['agents']=[]
        attribute['agents'].append(agents[i])
        coalitions.append(attribute)
    return coalitions

def mediator_func(two_coalitions,weighted_agg):
    proposal_text_GPT=[]
    proposal_embbeding=[]
    propos1=embed_sentence(two_coalitions[0]['proposal'], nlp,stop_words)
    propos2=embed_sentence(two_coalitions[1]['proposal'], nlp,stop_words)
    #given two_coalitions[0]['proposal'] and two_coalitions[1]['proposal'] generate 10 different optional aggregated sentences 
    for i in range(len(proposal)):
        proposal_embbeding.append(embed_sentence(proposal_text_GPT[i], nlp,stop_words))
    minimum=0
    if weighted_agg:
        opt=np.mean([[element * len(two_coalitions[0]['agents']) for element in propos1],[element * len(two_coalitions[1]['agents']) for element in propos2]], axis=0)
    else:
        opt=np.mean([propos1,propos2], axis=0)
    for i in range(len(proposal_embbeding)):
        if sum(abs(a - b) for a, b in zip(opt, proposal_embbeding[minimum]))<sum(abs(a - b) for a, b in zip(opt, proposal_embbeding[i])):
            minimum=i
    return proposal_text_GPT[minimum]
        
                                                         
def coalition_formation(coalitions,rnd,weighted_agg,dis):
    if rnd:
        two_coalitions=propose_new_coalitions_random(coalitions)
    else:
        two_coalitions=propose_new_coalitions_minmax(coalitions)
    proposal=mediator_func(two_coalitions,weighted_agg)
    attribute={}
    attribute['proposal']=proposal
    attribute['agents']=[]
    if dis>0:
        coalition1_approve=0
        coalition2_approve=1
        for agent in two_coalitions[0]:
             coalition1_approve+=approve_proposal(agent,proposal)
        for agent in two_coalitions[1]:
             coalition2_approve+=approve_proposal(agent,proposal)
        if coalition1_approve/len(two_coalitions[0])>dis and coalition2_approve/len(two_coalitions[1])>dis:
            coalitions.remove(two_coalitions[0])
            coalitions.remove(two_coalitions[1])
            attribute['agents'].append(two_coalitions[0]['agents'])
            attribute['agents'].append(two_coalitions[1]['agents'])
            coalitions.append(attribute)
    else:
        for agent in two_coalitions[0]:
            if approve_proposal(agent,proposal)==1:
                attribute['agents'].append(agent)
                coalitions[coalitions.index(two_coalitions[0])]['agents'].remove(agent)
        for agent in two_coalitions[1]:
            if approve_proposal(agent,proposal)==1:
                attribute['agents'].append(agent)
                coalitions[coalitions.index(two_coalitions[1])]['agents'].remove(agent)
        coalitions.append(attribute)
        for coalition in coalitions:
            if len(coalition['agents'])==0:
                coalitions.remove(coalition)
    return coalitions
                                                         
def Halt(coalition):
    flag = False
    for i in range(len(coalitions,num_agents)):
        if ((len(coalitions[i]['agents'])/num_agents)>0.5):
            flag=True
    return flag
                                                         
                                                         
# Main simulation
def main():
    num_agents = 20
    rnd=True
    dis=0
    weighted_agg=True
    agent=create_agents(num_agents)
    coalitions = initialize_coalitions(agents)
    while(Halt(coalitions,num_agents)==False):
        coalitions=coalition_formation(coalitions,rnd,weighted_agg,dis)


if __name__ == "__main__":
    main()
