import numpy as np
import networkx as nx
import pandas as pd
import random
import config

def generate_dag(nodes,prob):
    G=nx.gnp_random_graph(nodes,prob,directed=True)
    dag=nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
    print(nx.is_directed_acyclic_graph(dag))
    return dag

def gaussian(dag,samples):
    nodes=list(dag.nodes)
    num_nodes=len(nodes)
    tsort=list(nx.topological_sort(dag))
    data=np.zeros((samples,num_nodes))
    for node in tsort:
        parents=list(dag.predecessors(node))
        if not parents:
            #generate Gaussian distribution if no direct parent nodes found
            data[:,node]=np.random.normal(0,1,samples)
        else:
            #generate weights from parent nodes with Gaussian noise
            parentvals=data[:,parents]
            weights=np.random.uniform(0.5,1.5,len(parents)) 
            data[:,node]=np.dot(parentvals,weights)+np.random.normal(0,0.5,samples)

    return pd.DataFrame(data, columns=[f"node{i}" for i in tsort])

datasets=[]
for numnodes in config.gen_nodes:
    temp={}
    dag=generate_dag(numnodes,config.prob)
    data=gaussian(dag,config.samples)
    temp={"nodes":numnodes,"dag":dag,"data":data}
    datasets.append(temp)



