#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd

from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting

from hydra._internal.config_loader_impl import ConfigLoaderImpl 
from hydra.core.override_parser.overrides_parser import (
    OverridesParser,
    create_functions,
)
parser = OverridesParser(create_functions())

import sys
sys.path.insert(0,'..')
from train import main


# In[ ]:


def map_params_to_arg_list(params):
    """Method to map a dictionary of params to a list of string arguments"""
    arg_list = []
    for key in params:
        arg_list.append(str(key) + "=" + str(params[key]))
    return arg_list


# In[ ]:

# parameters to be optimized
opt_parameters = [
    {
        "name": "dataset.training.learning_rate",
        "type": "range",
        "bounds": [1e-6,1e-2],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "dataset.model4.eventLSTM.hidden_size",
        "type": "range",
        "bounds": [128,256],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "dataset.model4.frameLSTM.hidden_size",
        "type": "range",
        "bounds": [128,256],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "dataset.model4.attention_params.hidden_size",
        "type": "range",
        "bounds": [16,512],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
]


# In[ ]:


ax_client = AxClient()

ax_client.create_experiment(
    name="sbu-model4",
    parameters=opt_parameters,
    objective_name="do_train",
    minimize=False,  # Optional, defaults to False.
#     parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
#     outcome_constraints=["l2norm <= 1.25"],  # Optional.
)


# In[ ]:


hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="../configs")
CFG = hydra.compose("config.yaml")

# print(OmegaConf.to_yaml(CFG, resolve=True))


# In[ ]:


def do_train(parameters):
    ovrr = map_params_to_arg_list(parameters)
    ovrr = parser.parse_overrides(ovrr)
        
    ConfigLoaderImpl._apply_overrides_to_config(ovrr, CFG)
    print(OmegaConf.to_yaml(CFG, resolve=True))
    
    return main(CFG)


# In[ ]:


for i in range(20):
    curr_params, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=do_train(curr_params))



best_parameters, values = ax_client.get_best_parameters()
print(f'BEST PARAMETERS : {best_parameters}')

