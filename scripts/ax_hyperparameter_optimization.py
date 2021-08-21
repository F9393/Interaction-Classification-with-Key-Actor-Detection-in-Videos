from omegaconf import OmegaConf
from ax.service.ax_client import AxClient
from key_actor_detection.train import train
import sys

# parameters to be optimized
opt_parameters = [
    {
        "name": "training.learning_rate",
        "type": "range",
        "bounds": [1e-6,1e-2],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model4.eventLSTM.hidden_size",
        "type": "range",
        "bounds": [128,256],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model4.frameLSTM.hidden_size",
        "type": "range",
        "bounds": [128,256],
        "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        "log_scale": False,  # Optional, defaults to False.
    },
    {
        "name": "model4.attention_params.hidden_size",
        "type": "range",
        "bounds": [16,512],
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
    ovrr = map_params_to_arg_list(parameters)
    CFG = OmegaConf.merge(CFG, OmegaConf.from_dotlist(ovrr))
    print(OmegaConf.to_yaml(CFG, resolve=True))
    return train(CFG)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        raise Exception("Please pass path to config file!")

    CFG = OmegaConf.load(sys.argv[1])

    ax_client = AxClient()
    ax_client.create_experiment(
        name="sbu-model4",
        parameters=opt_parameters,
        objective_name="do_train",
        minimize=False,  # Optional, defaults to False.
    #     parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    #     outcome_constraints=["l2norm <= 1.25"],  # Optional.
    )

    for i in range(3):
        curr_params, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=do_train(CFG, curr_params))

    best_parameters, values = ax_client.get_best_parameters()
    print(f'BEST PARAMETERS : {best_parameters}')
