from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OLD_ROOT = Path(
    "/home/zxwang/repos/apricot_human_in_domain/egoverse_zarr_apricot_in_domain_min90_ego_view_right_arm_landscape"
)
DEFAULT_PART2_ROOT = Path(
    "/home/zxwang/repos/apricot_human_in_domain/egoverse_zarr_apricot_in_domain_part2_min90_ego_view_right_arm"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zxwang/repos/apricot_human_in_domain/egoverse_zarr_apricot_in_domain_merged_min90_ego_view_right_arm_landscape"
)


def _zarr_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.endswith(".zarr")
    )


def build_merged_root(*, roots: list[Path], output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    created = 0
    existing = 0
    collisions: list[dict[str, str]] = []

    for root in roots:
        for source in _zarr_dirs(root):
            link = output_root / source.name
            if link.exists() or link.is_symlink():
                try:
                    if link.resolve() == source.resolve():
                        existing += 1
                        continue
                except OSError:
                    pass
                collisions.append({"link": str(link), "source": str(source)})
                continue
            link.symlink_to(source, target_is_directory=True)
            created += 1

    if collisions:
        raise RuntimeError(
            f"Refusing to overwrite {len(collisions)} existing merged-root entries: {collisions[:5]}"
        )

    return {
        "output_root": str(output_root),
        "source_roots": [str(root) for root in roots],
        "created_links": int(created),
        "existing_links": int(existing),
        "total_links": int(len(_zarr_dirs(output_root))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a symlink-only merged apricot human Zarr root.")
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--part2-root", type=Path, default=DEFAULT_PART2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    summary = build_merged_root(
        roots=[args.old_root.expanduser(), args.part2_root.expanduser()],
        output_root=args.output_root.expanduser(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
