from omegaconf import OmegaConf
import sys
from key_actor_detection.train import train

def main():

    if len(sys.argv) == 1:
        raise Exception("Please pass path to config file!")

    CFG = OmegaConf.load(sys.argv[1])
    cli_cfg = OmegaConf.from_cli(sys.argv[2:])
    
    CFG = OmegaConf.merge(CFG, cli_cfg)
    print(OmegaConf.to_yaml(CFG))  

    return train(CFG)


if __name__ == "__main__":
    main()


#model3: BEST PARAMETERS : {'training.learning_rate': 0.0001, 'model3.eventLSTM.hidden_size': 512, 'model3.attention_type': 2, 'model3.attention_params.hidden_size': 64}
#model3: BEST PARAMETERS : {'training.learning_rate': 0.001, 'model3.eventLSTM.hidden_size': 512, 'model3.attention_type': 2, 'model3.attention_params.hidden_size': 256, 'model3.attention_params.bias': True}

#BEST PARAMETERS : {'training.wd': 0.6753641995468647, 'training.learning_rate': 0.001, 'model3.eventLSTM.hidden_size': 512, 'model3.attention_type': 2, 'model3.attention_params.hidden_size': 512}
#model4 with 61% test accuracy 150 epochs {'training.learning_rate': 1e-05, 'model4.frameLSTM.hidden_size': 512, 'model4.eventLSTM.hidden_size': 512, 'model4.attention_type': 1, 'model4.attention_params.hidden_s
#ize': 512, 'model4.attention_params.bias': True}