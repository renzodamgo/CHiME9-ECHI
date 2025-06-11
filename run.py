# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.dummy_inference import dummy_inference as inference
from scripts.evaluate import evaluate
from scripts.prepare import prepare


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.prepare.run:
        prepare(cfg.prepare)

    if cfg.inference.run:
        inference(cfg.inference)

    if cfg.evaluate.run:
        evaluate(cfg.evaluate)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
