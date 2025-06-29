# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.train.unpack import unpack
from scripts.train.train import run


@hydra.main(version_base=None, config_path="config/train", config_name="main_aria")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.unpack.run:
        unpack(cfg.unpack)

    if cfg.train.run:
        run(
            cfg.dataloading,
            cfg.model,
            cfg.train,
            cfg.base_dir,
            cfg.debug,
            cfg.wandb.entity,
            cfg.wandb.project,
        )


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
