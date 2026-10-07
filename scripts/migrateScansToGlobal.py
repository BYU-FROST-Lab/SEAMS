#!/usr/bin/env python3
"""Move legacy per-session scans into the global scans store.

Legacy sessions saved their point clouds as ``{session}/scans/{char}/{index}.pcd``
(or ``{session}/../scans/...`` when there were policy sub-directories). The global
store keeps one copy of each scan as ``{scans_dir}/{dataset}/{mission}/{timestamp_us}.pcd``.

For every session directory (a directory with ``timestamps.txt``) found under ``--root``:

* the dataset is read from the session's ``metadata.yaml``
* each symbol character is mapped to a mission from the directory name
  (``newMission__...__firstMission``). Tokens that are not missions of the dataset
  (no bag found) are dropped, e.g. ``approx__quad-hard__quad-easy`` -> A: quad-easy, B: quad-hard
* each scan is linked (or copied if linking fails) to the global store, skipping scans that already exist

Nothing is deleted unless ``--delete-legacy`` is given, and then only legacy ``scans/``
directories whose sessions were all migrated without problems are removed.

Example
-------
    python -m scripts.migrateScansToGlobal --root results/accuracyEval --dry-run
    python -m scripts.migrateScansToGlobal --root results
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gtsam

moduleDir = Path(__file__).resolve().parents[1]
if str(moduleDir) not in sys.path:
    sys.path.insert(0, str(moduleDir))

import common.fgHelpers as fgtools
from common.datasetPaths import isKnownMission


@dataclass
class SessionResult:
    sessionDir: Path
    legacyDir: Path | None = None
    linked: int = 0
    existing: int = 0
    sizeMismatch: int = 0
    missing: list[str] = field(default_factory=list)
    skipReason: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipReason is None and not self.missing


def parseMissionMap(text: str | None) -> dict[str, str] | None:
    if not text:
        return None
    mapping = {}
    for item in text.split(","):
        char, mission = item.split("=")
        mapping[char.strip()] = mission.strip()
    return mapping


def findSessionDirs(root: Path, scansDir: Path) -> list[Path]:
    sessions = []
    for timestampsFile in sorted(root.rglob("timestamps.txt")):
        sessionDir = timestampsFile.parent
        if scansDir in sessionDir.parents:
            continue
        sessions.append(sessionDir)
    return sessions


def sessionSymbolMap(sessionDir: Path, dataset: str) -> dict[str, str]:
    missions = [m for m in fgtools.missionsFromSessionDir(sessionDir) if isKnownMission(dataset, m)]
    return {chr(ord('A') + i): m for i, m in enumerate(missions)}


def placeScan(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.tmp.pcd")
    try:
        # A hard link takes no extra space and keeps the scan if the legacy directory is deleted later
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def migrateSession(sessionDir: Path, scansDir: Path, missionMap: dict[str, str] | None, dryRun: bool,
                   planned: set[Path] | None = None) -> SessionResult:
    '''
    :param planned: In a dry run, destinations that earlier sessions would have added, so scans shared between
        sessions are only counted once.
    '''
    res = SessionResult(sessionDir)

    res.legacyDir = fgtools.findLegacyScansDir(sessionDir)
    if res.legacyDir is None:
        res.skipReason = "no legacy scans directory"
        return res

    dataset = fgtools.readSessionDataset(sessionDir)
    if dataset is None:
        res.skipReason = "no dataset in metadata.yaml"
        return res

    symbolToMission = missionMap if missionMap is not None else sessionSymbolMap(sessionDir, dataset)
    if not symbolToMission:
        res.skipReason = f"no missions of dataset '{dataset}' found in directory name (use --mission-map)"
        return res

    import yaml
    with open(sessionDir / "metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f) or {}
    metadataMission = metadata.get("mission")
    newestMission = symbolToMission[max(symbolToMission.keys())]
    if metadataMission is not None and metadataMission != newestMission:
        res.skipReason = f"newest mission from directory name '{newestMission}' does not match metadata mission '{metadataMission}' (use --mission-map)"
        return res
    # Current session keys that were kept in the saved session belong to the newest mission
    symbolToMission = {**symbolToMission, metadata.get("currSessionSymbolChr", "c"): newestMission}

    timestamps = fgtools.loadTimestampsWithKeys(sessionDir / "timestamps.txt")
    keyChars = sorted(set(chr(gtsam.Symbol(k).chr()) for k in timestamps))
    if not set(keyChars).issubset(symbolToMission.keys()):
        res.skipReason = f"keys use symbols {keyChars} but missions are only known for {symbolToMission}"
        return res

    for key, t in timestamps.items():
        s = gtsam.Symbol(key)
        char = chr(s.chr())
        src = res.legacyDir / char / f"{s.index()}.pcd"
        dst = fgtools.globalScanPath(scansDir, dataset, symbolToMission[char], t)
        if dst.exists() or (planned is not None and dst in planned):
            res.existing += 1
            if dst.exists() and src.exists() and src.stat().st_size != dst.stat().st_size:
                res.sizeMismatch += 1
            continue
        if not src.exists():
            res.missing.append(s.string())
            continue
        if dryRun:
            if planned is not None:
                planned.add(dst)
        else:
            placeScan(src, dst)
        res.linked += 1

    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=moduleDir / "results", help="Directory searched recursively for sessions (default: results/)")
    parser.add_argument("--scans_dir", type=Path, default=moduleDir / "results" / "scans", help="Global scans directory (default: results/scans)")
    parser.add_argument("--mission-map", type=str, default=None,
                        help="Override the symbol to mission mapping for every session found, e.g. 'A=quad-easy,B=quad-hard'. "
                             "Use with --root pointing at a single session.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be done")
    parser.add_argument("--delete-legacy", action="store_true",
                        help="Delete legacy scans/ directories whose sessions were all migrated without problems")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    scansDir = args.scans_dir.resolve()
    missionMap = parseMissionMap(args.mission_map)

    sessions = findSessionDirs(root, scansDir)
    print(f"Found {len(sessions)} sessions under {root}")
    if args.dry_run:
        print("Dry run: nothing will be written or deleted")

    results = []
    planned: set[Path] = set()
    for sessionDir in sessions:
        res = migrateSession(sessionDir, scansDir, missionMap, args.dry_run, planned)
        results.append(res)
        rel = sessionDir.relative_to(root) if root in sessionDir.parents else sessionDir
        if res.skipReason:
            print(f"SKIP  {rel}: {res.skipReason}")
            continue
        status = "OK   " if res.ok else "WARN "
        msg = f"{status} {rel}: {'would add' if args.dry_run else 'added'} {res.linked}, already existed {res.existing}"
        if res.sizeMismatch:
            msg += f", {res.sizeMismatch} existing scans differ in size from the legacy scan (kept existing)"
        if res.missing:
            msg += f", {len(res.missing)} missing legacy scans (first: {res.missing[:5]})"
        print(msg)

    migrated = [r for r in results if r.skipReason is None]
    print(f"\n{len(migrated)} sessions migrated, {len(results) - len(migrated)} skipped, "
          f"{sum(r.linked for r in migrated)} scans {'would be ' if args.dry_run else ''}added")

    # Group sessions by the legacy directory they use. A shared {session}/scans is used by every policy sub-directory.
    legacyDirs: dict[Path, list[SessionResult]] = {}
    for r in results:
        if r.legacyDir is not None:
            legacyDirs.setdefault(r.legacyDir, []).append(r)
    # Only delete directories inside --root so sessions outside it that share a scans directory are never affected
    canDelete = lambda d, rs: all(r.ok for r in rs) and root in d.parents
    deletable = sorted(d for d, rs in legacyDirs.items() if canDelete(d, rs))
    blocked = sorted(d for d, rs in legacyDirs.items() if not canDelete(d, rs))

    if args.delete_legacy:
        print(f"\n{len(deletable)} legacy scans directories can be deleted, {len(blocked)} are kept because a session using them was skipped or had problems, or they are outside --root")
        for d in deletable:
            if args.dry_run:
                print(f"Would delete {d}")
            else:
                print(f"Deleting {d}")
                shutil.rmtree(d)
        for d in blocked:
            print(f"Keeping {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
