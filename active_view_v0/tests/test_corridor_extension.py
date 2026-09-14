from active_view_v0.corridors import extend_route_straight


class Rotation:
    def __init__(self, yaw):
        self.yaw = yaw


class Location:
    def __init__(self, x, y=0.0):
        self.x = x
        self.y = y
        self.z = 0.0

    def distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)


class Transform:
    def __init__(self, x, yaw):
        self.location = Location(x)
        self.rotation = Rotation(yaw)


class Waypoint:
    def __init__(self, x, yaw=0.0):
        self.transform = Transform(x, yaw)
        self.options = []

    def next(self, _distance):
        return self.options


def test_extend_route_prefers_straight_successor():
    start = Waypoint(0.0)
    turn = Waypoint(2.0, 90.0)
    straight = Waypoint(2.0, 0.0)
    end = Waypoint(4.0, 0.0)
    start.options = [turn, straight]
    straight.options = [end]
    result = extend_route_straight([start], 4.0, step_m=2.0)
    assert result == [start, straight, end]
