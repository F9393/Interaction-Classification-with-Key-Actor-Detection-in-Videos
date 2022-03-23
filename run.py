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
