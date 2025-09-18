#ground truth datasets
sets={"cdt":[["dream4-1","dream4-2","dream4-3","dream4-4","dream4-5"],"sachs"],
          "bnlearn": ["asia", "andes", "alarm"]}

#gaussian data generation
gen_nodes=[10,40,100]
prob=0.5
samples=5000

config = {
    "common": {
        "edge_probability": 0.5,       
    },
    "method_linear": {
        "node_counts": [10,40,70,100],  
        "num_samples": 5000,           
        "noise_types": ["lingam", "gaussian"], 
        "permutate": True,            
    },
    "method_2nd_order": {
        "node_counts": [10,40,70,100], 
        "num_samples": 1000,           
        "noise_types": ["lingam", "gaussian"],       
        "permutate": True,            
    },
    "method_gp": {
        "node_counts": [10],   
        "num_samples": 1000,           
        "noise_variance_range": (0.5, 1.5), 
    },
}
