config = {
    "causal_llm": {
        "input_dim": None,  
        "output_dim": lambda input_dim: input_dim * input_dim,
        "model_path": "llm_model.pth"
    },
    "RL": {
        "nb_epoch": 100
    },
    "ICALiNGAM": {
        "max_iter": 10000,
        "thresh": 0.1
    },
    "GraNDAG": {
        "input_dim": None, 
        "iterations": 1000
         
    },
    "GES": {},
    "PC": {}
}
