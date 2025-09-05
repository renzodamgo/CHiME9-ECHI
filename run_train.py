# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.train.unpack import unpack
from scripts.train.train_script_multi import run


@hydra.main(version_base=None, config_path="config/train", config_name="main_ha")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.unpack.run:
        print(cfg.unpack)
        unpack(cfg.unpack)

    if cfg.train.run:
        # Support resuming from checkpoint via config
        resume_checkpoint = None
        if hasattr(cfg.train, 'resume_from_checkpoint') and cfg.train.resume_from_checkpoint:
            resume_checkpoint = cfg.train.resume_from_checkpoint
            logging.info(f"🔄 Resuming training from checkpoint: {resume_checkpoint}")
        
        run(
            cfg.dataloading,
            cfg.model,
            cfg.train,
            cfg.train_dir,
            cfg.debug,
            cfg.wandb.entity,
            cfg.wandb.project,
            resume_checkpoint,
        )


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
