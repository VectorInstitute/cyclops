"""Draw system level evaluation framework architecture diagram."""

# pylint: disable=pointless-statement,expression-not-assigned

from os.path import join
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.programming.language import Python
from diagrams.programming.flowchart import Database
from diagrams.generic.database import SQL


ASSETS_DIR = "./assets"


def draw_implementation_on_gemini():
    """Draw system diagram for implementation on GEMINI."""
    with Diagram(
        "Evaluation framework on GEMINI",
        filename="evaluation_framework_on_gemini",
        show=False,
    ):
        with Cluster("dev environment"):
            jupyter = Custom("Jupyter", join(ASSETS_DIR, "jupyter.png"))
            _ = Custom("Gitlab", join(ASSETS_DIR, "gitlab.png"))
            _ = Python("Python")
        luigi = Custom("Luigi", join(ASSETS_DIR, "luigi.png"))

        with Cluster("drift detection dependencies"):
            evidently = Custom("Evidently", join(ASSETS_DIR, "evidently.png"))
            alibi = Custom("Alibi", join(ASSETS_DIR, "alibi.png"))
        mlflow = Custom(
            "Model registry/ Experiment tracking", join(ASSETS_DIR, "mlflow.png")
        )
        evidently_report = Custom("report", join(ASSETS_DIR, "evidently_report.png"))

        jupyter >> luigi >> [evidently, alibi]
        evidently >> evidently_report
        luigi >> mlflow


def draw_implementation_extended():
    """Draw system diagram for implementation with extended capabilities."""
    with Diagram(
        "Evaluation framework extended",
        filename="evaluation_framework_extended",
        show=False,
    ):
        with Cluster("dev environment"):
            jupyter = Custom("Jupyter", join(ASSETS_DIR, "jupyter.png"))
            _ = Custom("Gitlab", join(ASSETS_DIR, "gitlab.png"))
            _ = Python("Python")
        prefect = Custom("Prefect", join(ASSETS_DIR, "prefect.png"))

        with Cluster("drift detection dependencies"):
            evidently = Custom("Evidently", join(ASSETS_DIR, "evidently.png"))
            alibi = Custom("Alibi", join(ASSETS_DIR, "alibi.png"))
        mlflow = Custom(
            "Model registry/ Experiment tracking", join(ASSETS_DIR, "mlflow.png")
        )
        evidently_report = Custom("report", join(ASSETS_DIR, "evidently_report.png"))
        mlflow_dashboard = Custom(
            "MLFlow dashboard", join(ASSETS_DIR, "mlflow_dashboard.png")
        )
        prefect_dashboard = Custom(
            "Prefect dashboard", join(ASSETS_DIR, "prefect_dashboard.png")
        )

        jupyter >> prefect >> [evidently, alibi]
        evidently >> evidently_report
        prefect >> mlflow >> mlflow_dashboard
        prefect >> prefect_dashboard


def draw_dr_cyclops():
    """Draw design of components of dr_cyclops."""
    with Diagram("dr_cyclops", filename="dr_cyclops", show=False):
        with Cluster("hospital site A"):
            hosp_a = Custom("", join(ASSETS_DIR, "hospAB.png"))
            db_a = Custom("", join(ASSETS_DIR, "dbA.png"))
        with Cluster("hospital site B"):
            hosp_b = Custom("", join(ASSETS_DIR, "hospAB.png"))
            db_b = Custom("", join(ASSETS_DIR, "dbB.png"))
        with Cluster("hospital site Z"):
            _ = Custom("", join(ASSETS_DIR, "hospZ.png"))
            db_z = Custom("", join(ASSETS_DIR, "dbZ.png"))
        storage = Database("Databases (GEMINI/MIMIC)")
        stream = Custom("", join(ASSETS_DIR, "stream.png"))
        with Cluster("dr_cyclops data layer"):
            extract = SQL("extract")
            process = Custom("processing", join(ASSETS_DIR, "process.png"))
            feast = Custom("feature store", join(ASSETS_DIR, "feast_logo.png"))

        with Cluster("dr_cyclops model layer"):
            model_tr = Custom("model training", join(ASSETS_DIR, "model.png"))
            model_ev = Custom("model evaluation", join(ASSETS_DIR, "eval.png"))
        with Cluster("dr_cyclops drift-detection layer"):
            drift = Custom("drift detection", join(ASSETS_DIR, "drift.png"))

        db_a >> storage
        db_b >> storage
        db_z >> stream >> process
        hosp_a - Edge(color="brown", style="dotted") - hosp_b
        storage >> extract >> process >> feast
        feast >> model_tr
        feast >> model_ev
        feast >> drift


if __name__ == "__main__":
    draw_dr_cyclops()
    # draw_implementation_on_gemini()
    # draw_implementation_extended()
