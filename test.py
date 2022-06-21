from feast import FeatureStore


import matplotlib.pyplot as plt
import numpy as np

refuel_km = np.array([0, 505.4, 1070, 1690])
refuel_cost = np.array([40.1, 50, 63, 55])

carwash_km = np.array([302.0, 605.4, 901, 1331, 1788.2])
carwash_cost = np.array([35.0, 40.0, 35.0, 35.0, 35.0])

repair_km = np.array([788.0, 1605.4])
repair_cost = np.array([135.0, 74.5])

fig, ax = plt.subplots(figsize=(12, 3))

plt.scatter(
    refuel_km,
    np.full_like(refuel_km, 0),
    marker="o",
    s=100,
    color="lime",
    edgecolors="black",
    zorder=3,
    label="refuel",
)
plt.bar(
    refuel_km,
    refuel_cost,
    bottom=15,
    color="lime",
    ec="black",
    width=20,
    label="refuel cost",
)

plt.scatter(
    carwash_km,
    np.full_like(carwash_km, 0),
    marker="d",
    s=100,
    color="tomato",
    edgecolors="black",
    zorder=3,
    label="car wash",
)
plt.bar(
    carwash_km,
    -carwash_cost,
    bottom=-15,
    color="tomato",
    ec="black",
    width=20,
    label="car wash cost",
)

plt.scatter(
    repair_km,
    np.full_like(repair_km, 0),
    marker="^",
    s=100,
    color="lightblue",
    edgecolors="black",
    zorder=3,
    label="car repair",
)
# plt.bar(repair_km, -repair_cost, bottom=-15, color='lightblue', ec='black', width=20)

ax.spines["bottom"].set_position("zero")
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["left"].set_color("none")
ax.tick_params(axis="x", length=20)
ax.set_yticks([])  # turn off the yticks

_, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(-15, xmax)
ax.set_ylim(ymin, ymax + 25)  # make room for the legend
ax.text(xmax, -5, "km", ha="right", va="top", size=14)
plt.legend(ncol=5, loc="upper left")

plt.tight_layout()
plt.show()

fs = FeatureStore(repo_path="feature_repo")


data_pipeline = Pipeline(
    PipelineComponent(c) for c in [aggregator, statics_imputer, temporal_imputer]
)
