#!/usr/bin/env python3
"""Domain partition shared across the param-config paper's experiment scripts
(anytime-param-config.py and its rehab / ana-baseline siblings), so the shard
boundaries used for merge-id disjointness can't drift between them.

A random (seed=20260715) shuffle of the 42 agile-strips domains split into
sizes 11/11/10/10, so hard domains spread roughly evenly across shards rather
than clustering alphabetically. Edit freely, but they must stay a disjoint
partition of the discovered benchmark domains -- check_partition() verifies
this against whatever the caller discovers on disk. To regenerate:
    python -c "import os,random; d=sorted(...); random.Random(20260715).shuffle(d)"
"""

DOMAIN_SHARDS = {
    "shard1": [
        "agricola", "airport", "data-network", "floortile", "nomystery",
        "parking", "pegsol", "pipesworld-tankage", "sokoban", "tetris",
        "woodworking",
    ],
    "shard2": [
        "blocksworld", "elevators", "mprime", "pathways",
        "pipesworld-notankage", "satellite", "termes", "thoughtful",
        "tidybot", "tpp", "zenotravel",
    ],
    "shard3": [
        "barman", "depots", "freecell", "miconic", "openstacks",
        "organic-synthesis-split", "rovers", "snake", "storage", "visitall",
    ],
    "shard4": [
        "childsnack", "driverlog", "ged", "grid", "gripper", "hiking",
        "logistics", "parcprinter", "scanalyzer", "transport",
    ],
}
SHARD_NAMES = sorted(DOMAIN_SHARDS)


def check_partition(all_domains):
    """Return (dupes, unassigned, unknown) diagnosing whether DOMAIN_SHARDS is
    a clean partition of all_domains (each a sorted list; empty if clean)."""
    shard_domains = [d for ds in DOMAIN_SHARDS.values() for d in ds]
    dupes = sorted({d for d in shard_domains if shard_domains.count(d) > 1})
    unassigned = sorted(set(all_domains) - set(shard_domains))
    unknown = sorted(set(shard_domains) - set(all_domains))
    return dupes, unassigned, unknown
