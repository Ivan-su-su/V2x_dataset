"""Offline checks for reproducible scene/weather/run configuration."""

from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest

import yaml

from active_view_v0.config import load_config
from active_view_v0.scene_recipe import materialize
from active_view_v0.weather import validate_weather, weather_parameters


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "configs" / "dense_dynamic_town03_40s.yaml"
WEATHER = ROOT / "configs" / "weather" / "foggy_noon.yaml"
PROFILE = ROOT / "configs" / "profiles" / "offline_counterfactual.yaml"


class SceneRecipeTest(unittest.TestCase):
    def test_weather_variant_keeps_traffic_and_sensor_geometry(self) -> None:
        base = load_config(SCENE)
        built = materialize(SCENE, WEATHER, PROFILE, run_name="town03_fog_001")
        self.assertEqual(built["regional"]["support_vehicles"], base["regional"]["support_vehicles"])
        self.assertEqual(built["regional"]["fixed_signal_plan"], base["regional"]["fixed_signal_plan"])
        self.assertEqual(built["sensors"]["vehicle_lidar"], base["sensors"]["vehicle_lidar"])
        self.assertEqual(built["carla"]["seed"], base["carla"]["seed"])
        self.assertEqual(built["carla"]["weather"]["overrides"]["fog_density"], 60)
        self.assertEqual(built["output"]["run_name"], "town03_fog_001")
        self.assertEqual(len(built["recipe"]["scene"]["sha256"]), 64)

    def test_weather_and_profile_cannot_rewrite_traffic(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text(yaml.safe_dump({"carla": {"seed": 10}}))
            with self.assertRaisesRegex(ValueError, "only carla.weather"):
                materialize(SCENE, path, PROFILE)
            path.write_text(yaml.safe_dump({"profile": "bad", "overrides": {
                "regional": {"support_vehicles": []}}}))
            with self.assertRaisesRegex(ValueError, "cannot modify"):
                materialize(SCENE, WEATHER, path)

    def test_weather_is_copied_from_preset(self) -> None:
        preset = types.SimpleNamespace(cloudiness=15.0, fog_density=0.0)

        class WeatherParameters:
            ClearNoon = preset

            def __init__(self):
                self.cloudiness = 0.0
                self.fog_density = 0.0

        built = weather_parameters({"carla": {"weather": {
            "preset": "ClearNoon", "overrides": {"fog_density": 60.0}
        }}}, types.SimpleNamespace(WeatherParameters=WeatherParameters))
        self.assertEqual(built.cloudiness, 15)
        self.assertEqual(built.fog_density, 60)
        self.assertEqual(preset.fog_density, 0)
        with self.assertRaisesRegex(ValueError, "unsupported CARLA weather parameter"):
            validate_weather({"preset": "ClearNoon", "overrides": {"vehicle_speed": 99}})


if __name__ == "__main__":
    unittest.main()
