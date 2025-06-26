# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.evaluation.dummy_enhance import dummy_enhance as enhance
from scripts.evaluation.evaluate import evaluate
from scripts.evaluation.prepare import prepare
from scripts.evaluation.report import report
from scripts.evaluation.setup import setup
from scripts.evaluation.validate import validate


@hydra.main(version_base=None, config_path="config/evaluation", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.setup.run:
        setup(cfg.setup)

    if cfg.enhance.run:
        enhance(cfg.enhance)

    if cfg.validate.run:
        status = validate(cfg.validate)
        if not status:
            # The validate function logs detailed errors, so we just exit with a
            # failure code.
            raise SystemExit(1)

    if cfg.prepare.run:
        prepare(cfg.prepare)

    if cfg.evaluate.run:
        evaluate(cfg.evaluate)

    if cfg.report.run:
        report(cfg.report)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
