"""Materialize one immutable, auditable run YAML from scene + weather + profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import dump_effective_config, load_config, validate_config


PROFILE_FIELDS = {
    "output": {"root", "run_name"},
    "regional": {"snapshot_times_s", "save_overhead_rgb", "annotated_bev"},
    "sensors": {"preview_cameras"},
}


def _yaml_object(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return obj


def _provenance(path: Path) -> dict[str, str]:
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def materialize(
    scene: str | Path, weather: str | Path, profile: str | Path,
    *, run_name: str | None = None,
) -> dict[str, Any]:
    """Compose one existing full scene YAML with restricted independent overlays."""
    scene, weather, profile = (Path(item).expanduser().resolve()
                               for item in (scene, weather, profile))
    config = copy.deepcopy(load_config(scene))
    weather_data = _yaml_object(weather)
    if set(weather_data) != {"carla"} or set(weather_data["carla"]) != {"weather"}:
        raise ValueError("weather overlay may change only carla.weather")
    config["carla"]["weather"] = weather_data["carla"]["weather"]

    profile_data = _yaml_object(profile)
    if set(profile_data) != {"profile", "overrides"}:
        raise ValueError("profile YAML needs only profile and overrides keys")
    if not isinstance(profile_data["profile"], str) or not profile_data["profile"]:
        raise ValueError("profile must be a non-empty name")
    overrides = profile_data["overrides"]
    if not isinstance(overrides, dict):
        raise ValueError("profile.overrides must be a mapping")
    for section, changes in overrides.items():
        if section not in PROFILE_FIELDS or not isinstance(changes, dict):
            raise ValueError(f"profile cannot modify {section}")
        forbidden = set(changes) - PROFILE_FIELDS[section]
        if forbidden:
            raise ValueError(f"profile cannot modify {section}.{sorted(forbidden)}")
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(config[section].get(key), dict):
                config[section][key].update(copy.deepcopy(value))
            else:
                config[section][key] = copy.deepcopy(value)
    if run_name is not None:
        if not run_name or "/" in run_name or "\\" in run_name or run_name in (".", ".."):
            raise ValueError("run_name must be a simple non-empty directory name")
        config["output"]["run_name"] = run_name
    config["recipe"] = {
        "schema_version": 1,
        "scene": _provenance(scene),
        "weather": _provenance(weather),
        "profile": {**_provenance(profile), "name": profile_data["profile"]},
    }
    validate_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen scene/weather/run YAML")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", type=str)
    args = parser.parse_args()
    cfg = materialize(args.scene, args.weather, args.profile, run_name=args.run_name)
    output = args.output.expanduser().resolve()
    if output in (args.scene.resolve(), args.weather.resolve(), args.profile.resolve()):
        raise ValueError("output must not overwrite a source YAML")
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_effective_config(cfg, output)
    print(f"Saved frozen YAML: {output}")


if __name__ == "__main__":
    main()
