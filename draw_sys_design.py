"""Draw system level evaluation framework architecture diagram."""

from os.path import join
from diagrams import Diagram, Cluster
from diagrams.custom import Custom
from diagrams.programming.language import Python


ASSETS_DIR = "./assets"


def draw_implementation_on_gemini():
    with Diagram(
        "evaluation framework on Gemini",
        filename="evaluation_framework_on_gemini",
        show=False,
    ):
        with Cluster("dev environment"):
            jupyter = Custom("Jupyter", join(ASSETS_DIR, "jupyter.png"))
            gitlab = Custom("Gitlab", join(ASSETS_DIR, "gitlab.png"))
            python = Python("Python")
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
    with Diagram(
        "evaluation framework extended",
        filename="evaluation_framework_extended",
        show=False,
    ):
        with Cluster("dev environment"):
            jupyter = Custom("Jupyter", join(ASSETS_DIR, "jupyter.png"))
            gitlab = Custom("Gitlab", join(ASSETS_DIR, "gitlab.png"))
            python = Python("Python")
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


if __name__ == "__main__":
    draw_implementation_on_gemini()
    # draw_implementation_extended()
