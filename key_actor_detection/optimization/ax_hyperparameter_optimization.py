from omegaconf import OmegaConf
from ax.service.ax_client import AxClient
from key_actor_detection.train import train
import sys
import torch
import os
import json
from pytorch_lightning import loggers as pl_loggers

num_trials = 100

# parameters to be optimized
opt_parameters = [
    {
        "name": "training.learning_rate",
        "type": "choice",
        "values": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "training.wd",
        "type": "range",
        "bounds": [1e-8, 1],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    # {
    #     "name": "model1.frameLSTM.hidden_size",
    #     "type": "choice",
    #     "values": [128, 256, 512],
    #     "value_type": "int",  # Optional, defaults to inference from type of "bounds".
    #     "log_scale": False,  # Optional, defaults to False.
    # },
    {
        "name": "model3.eventLSTM.hidden_size",
        "type": "choice",
        "values": [256, 512],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model3.attention_type",
        "type": "choice",
        "values": [1, 2],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model3.attention_params.hidden_size",
        "type": "choice",
        "values": [256, 512],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
]


def map_params_to_arg_list(params):
    """Method to map a dictionary of params to a list of string arguments"""
    arg_list = []
    for key in params:
        arg_list.append(str(key) + "=" + str(params[key]))
    return arg_list


def do_train(CFG, parameters):
    if torch.distributed.is_initialized():
        broadcasted_params = [parameters]
        torch.distributed.broadcast_object_list(broadcasted_params, src=0, group=None)
        parameters = broadcasted_params[0]
    ovrr = map_params_to_arg_list(parameters)
    CFG = OmegaConf.merge(CFG, OmegaConf.from_dotlist(ovrr))
    return train(CFG)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        raise Exception("Please pass path to config file!")

    CFG = OmegaConf.load(sys.argv[1])
    cli_cfg = OmegaConf.from_cli(sys.argv[2:])
    CFG = OmegaConf.merge(CFG, cli_cfg)

    ax_client = AxClient(verbose_logging=False)
    ax_client2 = AxClient(verbose_logging=False)

    ax_client.create_experiment(
        name="hockey-model4",
        parameters=opt_parameters,
        objective_name="do_train",
        minimize=False,
    )

    ax_client2.create_experiment(
        name="hockey-model4_2",
        parameters=opt_parameters,
        objective_name="do_train",
        minimize=False,
    )

    loop_count = 0
    if os.getenv("SLURM_PROCID", "0") == "0":
        for i in range(num_trials):
            curr_params, trial_index = ax_client.get_next_trial()
            print(f"CURRENT PARAMETERS : {curr_params}")
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=do_train(CFG, curr_params)
            )
        best_parameters, values = ax_client.get_best_parameters()
        print(f"BEST PARAMETERS : {best_parameters}")
        with open(
            os.path.join(CFG.training.save_dir, "best_parames.json"), "w"
        ) as outfile:
            json.dump(best_parameters, outfile)

        # CFG.training.save_dir
    else:
        for i in range(num_trials):
            if loop_count == 0:
                curr_params, trial_index = ax_client2.get_next_trial()
                loop_count += 1
            do_train(CFG, curr_params)
