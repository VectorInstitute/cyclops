"""Run experimenter with pre-configured parameters."""
import os
import pickle

import hydra
from drift_detector import (
    ClinicalShiftApplicator,
    DCTester,
    Detector,
    Experimenter,
    Reductor,
    SyntheticShiftApplicator,
    TSTester,
)
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from cyclops.monitor.utils import get_args, get_obj_from_str


@hydra.main(config_path="./configs")
def run_experiment(cfg: DictConfig):
    """Run experiment."""
    cfg = cfg[list(cfg.keys())[0]]

    dataset = get_obj_from_str(cfg.dataset.object)
    dataset_cfg = os.path.join(get_original_cwd(), cfg.dataset.cfg_path)
    x, metadata, metadata_mapping = dataset(dataset_cfg).get_data()

    reductor = Reductor(**get_args(Reductor, cfg.reductor))

    if cfg.tester.type == "TSTester":
        with open_dict(cfg):
            cfg.tester.pop("type")
        tester = TSTester(**cfg.tester)
    elif cfg.tester.type == "DCTester":
        with open_dict(cfg):
            cfg.tester.pop("type")
        tester = DCTester(**cfg.tester)

    detector = Detector(
        reductor=reductor,
        tester=tester,
        **cfg.detector,
    )
    if cfg.shiftapplicator is not None:
        if cfg.shiftapplicator.type == "SyntheticShiftApplicator":
            with open_dict(cfg):
                cfg.shiftapplicator.pop("type")
            shiftapplicator = SyntheticShiftApplicator(**cfg.shiftapplicator)
        elif cfg.shiftapplicator.type == "ClinicalShiftApplicator":
            with open_dict(cfg):
                cfg.shiftapplicator.pop("type")
            shiftapplicator = ClinicalShiftApplicator(**cfg.shiftapplicator)
    else:
        shiftapplicator = None

    experiment = Experimenter(
        **cfg.experimenter, detector=detector, shiftapplicator=shiftapplicator
    )

    results = experiment.run(x, metadata, metadata_mapping)
    with open(cfg.results_path, "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    run_experiment()
