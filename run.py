# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.dummy_enhance import dummy_enhance as enhance
from scripts.evaluate import evaluate
from scripts.prepare import prepare
from scripts.report import report


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.prepare.run:
        prepare(cfg.prepare)

    if cfg.enhance.run:
        enhance(cfg.enhance)

    if cfg.evaluate.run:
        evaluate(cfg.evaluate)

    if cfg.report.run:
        report(cfg.report)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
