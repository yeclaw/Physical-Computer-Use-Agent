"""Package-relative resource roots, independent of canonical module depth."""
from pathlib import Path


def package_root():
    return Path(__file__).resolve().parent


def source_root():
    return Path(__file__).resolve().parents[2]


def repository_root():
    return Path(__file__).resolve().parents[3]
