"""Validated CARLA weather shared by editor, preview and offline collection."""

from __future__ import annotations

import math
from typing import Any


PRESETS = frozenset({
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "SoftRainNoon",
    "MidRainyNoon", "HardRainNoon", "ClearSunset", "CloudySunset",
    "WetSunset", "WetCloudySunset", "SoftRainSunset", "MidRainSunset",
    "HardRainSunset",
})

# These are documented by the CARLA 0.9.16 WeatherParameters API.
RANGES = {
    "cloudiness": (0, 100), "precipitation": (0, 100),
    "precipitation_deposits": (0, 100), "wind_intensity": (0, 100),
    "sun_azimuth_angle": (0, 360), "sun_altitude_angle": (-90, 90),
    "fog_density": (0, 100), "fog_distance": (0, float("inf")),
    "wetness": (0, 100), "fog_falloff": (0, float("inf")),
    "scattering_intensity": (0, float("inf")),
    "mie_scattering_scale": (0, float("inf")),
    "rayleigh_scattering_scale": (0, float("inf")),
    "dust_storm": (0, 100),
}


def validate_weather(config: Any) -> None:
    if not isinstance(config, dict) or set(config) - {"preset", "overrides"}:
        raise ValueError("carla.weather must contain only preset and overrides")
    preset = config.get("preset", "ClearNoon")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError(f"unsupported CARLA weather preset: {preset}")
    overrides = config.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("carla.weather.overrides must be a mapping")
    for name, value in overrides.items():
        if name not in RANGES:
            raise ValueError(f"unsupported CARLA weather parameter: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"weather {name} must be numeric")
        lo, hi = RANGES[name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"weather {name} must be in [{lo}, {hi}]")


def weather_parameters(config: dict[str, Any], carla: Any) -> Any:
    """Copy a preset before overriding it: do not mutate CARLA's global preset."""
    spec = config.get("carla", {}).get("weather", {"preset": "ClearNoon"})
    validate_weather(spec)
    preset = getattr(carla.WeatherParameters, spec.get("preset", "ClearNoon"))
    weather = carla.WeatherParameters()
    for name in RANGES:
        if hasattr(preset, name) and hasattr(weather, name):
            setattr(weather, name, float(getattr(preset, name)))
    for name, value in spec.get("overrides", {}).items():
        if not hasattr(weather, name):
            raise ValueError(f"weather {name} is unavailable in this CARLA build")
        setattr(weather, name, float(value))
    return weather
