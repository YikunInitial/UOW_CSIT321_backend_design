"""Operational command line for status, backup, restore, and retention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from Agents.operations_service import OperationsService


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default=os.getenv("STEEL_API_RUN_ROOT", "runtime/api")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    backup = subparsers.add_parser("backup")
    backup.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify-backup")
    verify.add_argument("--archive", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--target", required=True)
    restore.add_argument("--confirm", required=True)
    prune = subparsers.add_parser("prune")
    prune.add_argument("--retention-days", type=int, default=90)
    prune.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    database = data_root / "state.db"
    if args.command == "verify-backup":
        _print(OperationsService.verify_backup(args.archive))
        return
    if args.command == "restore":
        if args.confirm != "RESTORE":
            raise SystemExit("Restore requires --confirm RESTORE")
        _print(OperationsService.restore_backup(args.archive, args.target))
        return
    service = OperationsService(data_root, database)
    if args.command == "status":
        _print(service.status())
    elif args.command == "backup":
        _print(service.create_backup(args.output))
    elif args.command == "prune":
        _print(
            service.prune_superseded_artifacts(
                retention_days=args.retention_days, apply=args.apply
            )
        )


if __name__ == "__main__":
    main()
