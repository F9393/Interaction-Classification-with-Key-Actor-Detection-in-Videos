'''
Model3 Ax:

pt_parameters = [
    {
        "name": "training.learning_rate",
        "type": "choice",
        "values": [1e-5,1e-4,1e-3,1e-2],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    # {
    #     "name": "model3.frameLSTM.hidden_size",
    #     "type": "choice",
    #     "values": [128, 256, 512],
    #     "value_type": "int",  # Optional, defaults to inference from type of "bounds".
    #     "log_scale": False,  # Optional, defaults to False.
    # },
    {
        "name": "model3.eventLSTM.hidden_size",
        "type": "choice",
        "values": [64,256,512],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model3.attention_type",
        "type": "choice",
        "values": [1,2],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model3.attention_params.hidden_size",
        "type": "choice",
        "values": [64,256,512],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model3.attention_params.bias",
        "type": "choice",
        "values": ['true', 'false'],
        "value_type": "bool",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
]

#model3: BEST PARAMETERS : {'training.learning_rate': 0.001, 'model3.eventLSTM.hidden_size': 512, 'model3.attention_type': 2, 'model3.attention_params.hidden_size': 256, 'model3.attention_params.bias': True}
[{'test_acc_step': 0.5769230723381042}]

'''

##########################################
