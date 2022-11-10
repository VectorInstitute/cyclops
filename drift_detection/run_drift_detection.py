
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.utils import get_original_cwd
from drift_detector import (
    Reductor,
    TSTester,
    DCTester,
    Detector,
    Experimenter,
    SyntheticShiftApplicator,
    ClinicalShiftApplicator,
)

from drift_detection.drift_detector.utils import get_obj_from_str, get_args
import pickle
import os

@hydra.main(config_path="./configs")
def run_experiment(cfg: DictConfig):
    """Run experiment."""

    cfg = cfg[list(cfg.keys())[0]]
    dataset = get_obj_from_str(cfg.dataset.object)
    dataset_cfg = os.path.join(get_original_cwd(), cfg.dataset.cfg_path)
    x, metadata, metadata_mapping = dataset(dataset_cfg).get_data()

    reductor = Reductor(
        **get_args(Reductor, cfg.reductor)
    )

    if cfg.tester.type == "TSTester":
        with open_dict(cfg):
            cfg.tester.pop("type")
        tester = TSTester(
            **cfg.tester
        )
    elif cfg.tester.type == "DCTester":
        with open_dict(cfg):
            cfg.tester.pop("type")
        tester = DCTester(
            **cfg.tester
        )
    
    detector = Detector(
        reductor=reductor,
        tester=tester,
        **cfg.detector,
    )
    if cfg.shiftapplicator is not None:
        if cfg.shiftapplicator.type == "SyntheticShiftApplicator":
            with open_dict(cfg):
                cfg.shiftapplicator.pop("type")
            shiftapplicator = SyntheticShiftApplicator(
                **cfg.shiftapplicator
            )
        elif cfg.shiftapplicator.type == "ClinicalShiftApplicator":
            with open_dict(cfg):
                cfg.shiftapplicator.pop("type")
            shiftapplicator = ClinicalShiftApplicator(
            **cfg.shiftapplicator
            )
    else:
        shiftapplicator = None

    experiment = Experimenter(
        **cfg.experimenter,
        detector=detector,
        shiftapplicator=shiftapplicator
    )

    results = experiment.run(x, metadata, metadata_mapping)
    print(os.getcwd())
    with open(cfg.results_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    results = run_experiment()

    