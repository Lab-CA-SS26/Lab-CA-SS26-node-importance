#!/usr/bin/env python3
"""Download raw graph datasets used in BRAVA-GNN experiments.

Default: 14 paper test graphs (9 from SNAP, 5 from the ABCDE release) into raw/ + abcde/.
Pass --calibration to also fetch the 10 parameter-tuning graphs used in the appendix.

Idempotent: skips files that already exist unless --force is passed.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
ABCDE = ROOT / "abcde"

TEST_SNAP = [
    ("https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz", "p2p-Gnutella31"),
    ("https://snap.stanford.edu/data/soc-Epinions1.txt.gz", "soc-Epinions1"),
    ("https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz", "soc-Slashdot0902"),
    ("https://snap.stanford.edu/data/email-EuAll.txt.gz", "email-EuAll"),
    ("https://snap.stanford.edu/data/web-Google.txt.gz", "web-Google"),
    ("https://snap.stanford.edu/data/wiki-Talk.txt.gz", "wiki-Talk"),
    ("https://snap.stanford.edu/data/wiki-topcats.txt.gz", "wiki-topcats"),
    ("https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz", "soc-Pokec"),
    ("https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz", "soc-LiveJournal1"),
]


# ABCDE release: ships 5 graphs (amazon, cit-Patents, com-lj, com-youtube, dblp) + BC scores.
ABCDE_ZIP = "https://github.com/MartinXPN/abcde/releases/download/v1.0.0/real.zip"

CAL_SNAP = [
    ("https://snap.stanford.edu/data/p2p-Gnutella30.txt.gz", "p2p-Gnutella30"),
    ("https://snap.stanford.edu/data/email-Enron.txt.gz", "email-Enron"),
    ("https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz", "soc-Slashdot0811"),
    ("https://snap.stanford.edu/data/web-NotreDame.txt.gz", "web-NotreDame"),
    ("https://snap.stanford.edu/data/web-BerkStan.txt.gz", "web-BerkStan"),
    ("https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz", "com-DBLP"),
    ("https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz", "com-Orkut"),
]

# (url, dest_name, archive_member_basename) — CSV archives converted to space-separated edgelist.
CAL_CSV = [
    ("https://snap.stanford.edu/data/git_web_ml.zip", "musae-github", "musae_git_edges.csv"),
    ("https://snap.stanford.edu/data/gemsec_facebook_dataset.tar.gz", "gemsec-Facebook", "artist_edges.csv"),
    ("https://snap.stanford.edu/data/twitch_gamers.zip", "twitch-gamers", "large_twitch_edges.csv"),
]


def _download(url: str, dest: Path) -> None:
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _strip_prefix(path: Path, prefix: str) -> None:
    """Rewrite file in place, dropping lines starting with prefix."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        pfx = prefix.encode()
        for line in src:
            if not line.startswith(pfx):
                dst.write(line)
    tmp.replace(path)


def fetch_snap(url: str, name: str, force: bool) -> None:
    dest = RAW / f"{name}.txt"
    if dest.exists() and not force:
        print(f"[skip] {name}")
        return
    print(f"[snap] {name}")
    with tempfile.TemporaryDirectory() as td:
        gz = Path(td) / "download.txt.gz"
        _download(url, gz)
        with gzip.open(gz, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    _strip_prefix(dest, "#")


def fetch_csv(url: str, name: str, member: str, force: bool) -> None:
    dest = RAW / f"{name}.txt"
    if dest.exists() and not force:
        print(f"[skip] {name}")
        return
    print(f"[csv] {name}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        archive = td / Path(url).name
        _download(url, archive)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(td)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(td)
        archive.unlink()
        matches = list(td.rglob(member))
        if not matches:
            raise FileNotFoundError(f"{member} not found in {url}")
        csv = matches[0]
        with open(csv) as src, open(dest, "w") as out:
            next(src, None)  # drop header
            for line in src:
                out.write(line.replace(",", " "))


def fetch_abcde(force: bool) -> None:
    expected = ["amazon", "cit-Patents", "com-lj", "com-youtube", "dblp"]
    missing = [g for g in expected if not (ABCDE / f"{g}.txt").exists() or not (ABCDE / f"{g}-score.txt").exists()]
    if not missing and not force:
        print("[skip] abcde bundle (all 5 graphs present)")
        return
    print(f"[abcde] fetching {ABCDE_ZIP}")
    ABCDE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zip_path = td / "real.zip"
        _download(ABCDE_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        # Archive structure: real/<graph>.txt, real/<graph>-score.txt
        extracted = next(td.rglob("amazon.txt")).parent
        for f in extracted.iterdir():
            shutil.move(f, ABCDE / f.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", action="store_true", help="Also fetch the 10 parameter-tuning graphs.")
    ap.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    print("=== Table 11 (test set) ===")
    for url, name in TEST_SNAP:
        fetch_snap(url, name, args.force)
    fetch_abcde(args.force)

    if args.calibration:
        print("=== Table 12 (calibration set) ===")
        for url, name in CAL_SNAP:
            fetch_snap(url, name, args.force)
        for url, name, member in CAL_CSV:
            fetch_csv(url, name, member, args.force)

    print("Done.")


if __name__ == "__main__":
    main()
