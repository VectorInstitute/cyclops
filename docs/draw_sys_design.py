"""Draw system level evaluation framework architecture diagram."""

# pylint: disable=pointless-statement,expression-not-assigned

from os.path import join

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.generic.database import SQL
from diagrams.programming.flowchart import Database
from diagrams.programming.language import Python

ASSETS_DIR = "./assets"


graph_attr = {
    "fontsize": "20",
}

fig_attr = {
    "fontsize": "32",
}


def draw_cyclops():
    """Draw design of components of cyclops."""
    with Diagram(
        "CyclOps Framework Design",
        filename="cyclops_design",
        show=False,
        graph_attr=fig_attr,
        node_attr=graph_attr,
    ):
        hosp_a = Custom("hospital site A", join(ASSETS_DIR, "hospAB.png"))
        hosp_b = Custom("hospital site B", join(ASSETS_DIR, "hospAB.png"))
        hosp_z = Custom("hospital site Z", join(ASSETS_DIR, "hospZ.png"))
        storage = Database("Retrospective data")
        with Cluster("CyclOps data pipeline", graph_attr=graph_attr):
            prefect = Custom("", join(ASSETS_DIR, "prefect.png"))
            with Cluster("CyclOps data modules", graph_attr=graph_attr):
                query = Custom("query API", join(ASSETS_DIR, "sqlalchemy.png"))
                process = Custom("processing API", join(ASSETS_DIR, "process.png"))
                feast = Custom("feature store", join(ASSETS_DIR, "feast_logo.png"))

        model_tr = Custom("model training", join(ASSETS_DIR, "model.png"))
        model_ev = Custom("model evaluation", join(ASSETS_DIR, "eval.png"))
        shift = Custom("shift detection", join(ASSETS_DIR, "drift.png"))

        visualizer = Custom("visualizer app", join(ASSETS_DIR, "visualizer.png"))

        fhir = Custom("", join(ASSETS_DIR, "fhir.png"))

        hosp_a >> storage
        hosp_b >> storage
        hosp_a - Edge(color="brown", style="dotted") - hosp_b
        hosp_z - Edge(color="red", style="dashed") - fhir
        fhir - Edge(color="red", style="dashed") - process
        storage >> query >> process >> feast
        feast >> model_tr
        feast >> model_ev
        feast >> shift
        feast >> visualizer
        model_ev >> visualizer
        shift >> visualizer


if __name__ == "__main__":
    draw_cyclops()
    # draw_implementation_on_gemini()
    # draw_implementation_extended()
