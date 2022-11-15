from src import model


def analyze_scene() -> model.World:
    # TODO dummy data
    return model.World(
        robot=model.Robot(
            position=model.Point(0, 0),
            angle=0,
        ),
        goal=model.Point(0, 0),
        obstacles=[],
    )
