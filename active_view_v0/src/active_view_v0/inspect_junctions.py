from __future__ import annotations

import argparse
import json
from pathlib import Path

from .junctions import rank_junctions


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank CARLA junctions for the active-view pilot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default=None, help="load this CARLA map before inspection")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.map and not world.get_map().name.endswith(args.map):
        world = client.load_world(args.map)
    infos = rank_junctions(world.get_map())
    records = [info.as_dict() for info in infos]
    print(f"Map: {world.get_map().name}; junctions: {len(records)}")
    print("rank  id      center(x,y,z)                 roads lanes straight extent(x,y)")
    for rank, record in enumerate(records[: args.top]):
        center = record["center"]
        extent = record["extent"]
        print(
            f"{rank:>4}  {record['junction_id']:<7} "
            f"({center[0]:>7.1f},{center[1]:>7.1f},{center[2]:>4.1f})  "
            f"{record['road_count']:>5} {record['lane_pair_count']:>5} "
            f"{record['straight_pair_count']:>8} ({extent[0]:.1f},{extent[1]:.1f})"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

