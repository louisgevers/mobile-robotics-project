from typing import Sequence
import pyvisgraph as vg
from src import model


def find_path(world: model.World) -> Sequence[model.Point]:
    # 1. Convert model points to point objects for the visibility graph library
    polygons = []
    for obstacle in world.obstacles:
        polygon = [convert_to_vg_point(point) for point in obstacle]
        polygons.append(polygon)
    # 2. Compute the visibility graph
    graph = vg.VisGraph()
    graph.build(polygons, status=False)
    # 3. Convert start and end points for the visibility graph library
    start = convert_to_vg_point(world.robot.position)
    end = convert_to_vg_point(world.goal)
    # 4. Compute shortest path
    path = graph.shortest_path(start, end)
    # 5. Convert back to model points
    return [convert_from_vg_point(point) for point in path]


def convert_to_vg_point(point: model.Point) -> vg.Point:
    return vg.Point(point.x, point.y)


def convert_from_vg_point(point: vg.Point) -> model.Point:
    return model.Point(point.x, point.y)
