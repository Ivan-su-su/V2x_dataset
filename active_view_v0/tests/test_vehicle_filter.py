from active_view_v0.regional_scenario import _safe_vehicle


class _Attribute:
    def __init__(self, value):
        self.value = value

    def as_int(self):
        return int(self.value)

    def __str__(self):
        # Reproduce a CARLA build that does not stringify to the raw value.
        return f"ActorAttribute(value={self.value})"


class _Blueprint:
    def __init__(self, blueprint_id, wheels):
        self.id = blueprint_id
        self.wheels = wheels

    def has_attribute(self, key):
        return key == "number_of_wheels"

    def get_attribute(self, key):
        assert key == "number_of_wheels"
        return _Attribute(self.wheels)


def test_vehicle_filter_uses_typed_actor_attribute_accessor():
    assert _safe_vehicle(_Blueprint("vehicle.tesla.model3", 4))
    assert not _safe_vehicle(_Blueprint("vehicle.bh.crossbike", 2))
    assert not _safe_vehicle(_Blueprint("walker.pedestrian.0001", 4))
