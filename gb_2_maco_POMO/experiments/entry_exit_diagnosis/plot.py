"""Route and edge-difference figures from saved coordinates and cycles."""

from __future__ import annotations

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from core import compare, segments


COLORS = {"A": "#d62728", "B": "#ff8c00", "C": "#6a3d9a"}


def draw(path, problem, spatial, failed, optimal, route_id, optimum, failed_length):
    cities = problem.cities
    balls = problem.city_to_ball
    _, removed, added, failed_classes, optimal_classes = compare(failed, optimal, balls)
    points = {city for edge in removed | added for city in edge}
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.subplots_adjust(left=.07, right=.97, bottom=.07, top=.90, wspace=.28, hspace=.34)

    def background(ax):
        for ball_id, ball in enumerate(problem.granular_balls):
            ax.add_patch(Circle((ball.center.x, ball.center.y), ball.radius,
                                fill=False, edgecolor="#708090", linewidth=1, linestyle="--", alpha=.65))
            ax.text(ball.center.x, ball.center.y, f"B{ball_id}", color="#34495e",
                    fontsize=8, ha="center", va="center")
        ax.scatter([c.x for c in cities], [c.y for c in cities], c=[spatial[c.number] for c in cities],
                   cmap="tab10", s=15, zorder=4)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def line(ax, edge, color, width=1.2, style="-"):
        a, b = edge
        ax.plot([cities[a].x, cities[b].x], [cities[a].y, cities[b].y],
                color=color, linewidth=width, linestyle=style, alpha=.85, zorder=2)

    for ax, route, title in ((axes[0, 0], optimal, f"Known optimal  {optimum}"),
                             (axes[0, 1], failed, f"Saved 2-opt failure  {failed_length}")):
        background(ax)
        for i in range(len(route)):
            line(ax, tuple(sorted((route[i], route[(i+1)%len(route)]))), "#777777", .7)
        for piece in segments(route, balls):
            for key, marker, color in (("entry", "^", "#0d6b58"), ("exit", "s", "#0054a6")):
                city = piece[key]
                ax.scatter([cities[city].x], [cities[city].y], marker=marker, c=color, s=45, zorder=6)
                ax.annotate(f"{key[0].upper()}{city}", (cities[city].x, cities[city].y),
                            xytext=(4, 4), textcoords="offset points", fontsize=6)
        ax.set_title(title)

    for ax in axes[1]:
        background(ax)
        for edge in sorted(added):
            line(ax, edge, COLORS[optimal_classes[edge]], 2.0, "--")
        for edge in sorted(removed):
            line(ax, edge, COLORS[failed_classes[edge]], 2.7)
        for city in sorted(points):
            ax.annotate(str(city), (cities[city].x, cities[city].y), xytext=(4,4),
                        textcoords="offset points", fontsize=8, color="black", zorder=8)
    axes[1, 0].set_title("Difference edges: solid=delete, dashed=add")
    affected = Counter(ball for edge in removed | added for city in edge for ball in [balls[city]])
    target_ball = affected.most_common(1)[0][0]
    local = [city for city in points if balls[city] == target_ball]
    xs, ys = [cities[i].x for i in local], [cities[i].y for i in local]
    padx, pady = max(30, (max(xs)-min(xs))*.25), max(30, (max(ys)-min(ys))*.25)
    axes[1, 1].set_xlim(min(xs)-padx, max(xs)+padx)
    axes[1, 1].set_ylim(min(ys)-pady, max(ys)+pady)
    axes[1, 1].set_title(f"Difference zoom: ball B{target_ball}")
    fig.suptitle(f"{route_id} | red=A cross-ball, orange=B interface-near, purple=C deep inside | circles=balls; dot color=spatial cluster", fontsize=10)
    fig.savefig(path, dpi=170)
    plt.close(fig)
