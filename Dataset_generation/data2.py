import pandas as pd
import csv
import numpy as np
import networkx as nx
import cdt
import bnlearn as bn
import config 

def create_dag(adj_mat,nodes): 
    #create dag given the adjacency matrix
    dag=nx.DiGraph()
    n=adj_mat.shape[0]
    for i in range(n):
      for j in range(n):
          if adj_mat[i][j]:
              dag.add_edge(i,j)

#return results as dictionary of format 
#{"data":<dataframe>, "nodes":<np.ndarray>, "adjacency_matrix":<np.ndarray>, "dag":<nx.digraph>}

#cdt (dream4-1,dream4-2,dream4-3,dream4-4,dream4-5,sachs)
def data_cdt(dataset):
    data, graph=cdt.data.load_dataset(dataset)
    nodes=np.array(graph.nodes())
    adjacency_matrix=nx.adjacency_matrix(graph).todense()
    return {"data":data,"nodes":nodes,"adjacency_matrix":adjacency_matrix,"dag": graph}

#bnlearn(asia,andes,alarm)
def data_bnlearn(dataset):
    df=bn.bnlearn.import_example(dataset)
    dag=bn.bnlearn.import_DAG(dataset)
    nodes=list(dag['adjmat'].columns)
    temp=dag['adjmat'].to_numpy()
    adj=np.zeros((len(nodes),len(nodes)),dtype=int)
    for i in range(len(temp)):
      for j in range(len(temp)):
        if temp[i,j]:
          adj[i,j]=1
    return {"data":df,"nodes":nodes,"adjacency matrix":adj,"dag":create_dag(adj,nodes)}


#cdt
class cdt_data:

    @staticmethod
    def dream4_all(): 
        #list of all dream4 dags and adj matrices
        return [data_cdt(i) for i in config.sets["cdt"][0]]

    @staticmethod
    def sachs():
        return data_cdt(config.sets["cdt"][1])

#bnlearn
class bnlearn_data:

    @staticmethod
    def asia():
        return data_bnlearn(config.sets["bnlearn"][0])

    @staticmethod
    def andes():
        return data_bnlearn(config.sets["bnlearn"][1])

    @staticmethod
    def alarm():
        return data_bnlearn(config.sets["bnlearn"][2])