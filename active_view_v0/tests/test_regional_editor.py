"""Offline checks for editor serialization and camera-to-map picking."""

from pathlib import Path
import unittest

import numpy as np

from active_view_v0.bev_render import project_world_to_camera
from active_view_v0.config import load_config
from active_view_v0.regional_editor import edited_config, pixel_to_ground


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dense_dynamic_town03_40s.yaml"


class RegionalEditorTest(unittest.TestCase):
    def test_edit_preserves_source_and_extra_vehicle_properties(self) -> None:
        cfg = load_config(CONFIG)
        first = cfg["regional"]["support_vehicles"][0]
        incoming = [{key: first[key] for key in
                     ("role", "route", "start_offset_m", "speed_difference_pct", "required")}]
        incoming[0]["start_offset_m"] = 66
        incoming.append({"role": "regional_extra_1", "route": "corridor_oncoming",
                         "start_offset_m": 80, "speed_difference_pct": 5, "required": False})
        edited = edited_config(cfg, {"support_vehicles": incoming, "j2_switch_time_s": 22})
        self.assertEqual(cfg["regional"]["fixed_signal_plan"]["j2_switch_time_s"], 20)
        self.assertEqual(edited["regional"]["fixed_signal_plan"]["j2_switch_time_s"], 22)
        self.assertEqual(len(edited["regional"]["support_vehicles"]), 2)
        self.assertEqual(edited["regional"]["support_vehicles"][0]["blueprints"], first["blueprints"])
        self.assertEqual(cfg["regional"]["support_vehicles"][0]["start_offset_m"], first["start_offset_m"])

    def test_reject_invalid_route(self) -> None:
        cfg = load_config(CONFIG)
        with self.assertRaisesRegex(ValueError, "unsupported route"):
            edited_config(cfg, {"support_vehicles": [{"role": "regional_extra_1",
                                                        "route": "sidewalk", "start_offset_m": 10}]})

    def test_pixel_projection_round_trip(self) -> None:
        transform = np.eye(4)
        transform[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        transform[:3, 3] = [30, -20, 100]
        location = pixel_to_ground((770, 400), transform, 1200, 1000, 90, 0)
        uv, valid = project_world_to_camera(location[None], np.linalg.inv(transform),
                                            1200, 1000, 90)
        self.assertTrue(bool(valid[0]))
        np.testing.assert_allclose(uv[0], [770, 400], atol=1e-6)
        self.assertAlmostEqual(location[2], 0)


if __name__ == "__main__":
    unittest.main()
