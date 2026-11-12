"""
Entry point: run one version of the tutorial from the repository root.

    python run.py 1        # version1 ... version6
    python run.py 6 --epochs 5 --activation tanh   # extra flags go to v5/v6

Each version lives in its own folder (versionN/) and is a complete snapshot,
so you can diff two consecutive folders to see exactly what was added.
"""
import argparse
import importlib

ENTRY_POINTS = {
    1: 'version1.backend',
    2: 'version2.backend',
    3: 'version3.backend',
    4: 'version4.backend',
    5: 'version5.main',
    6: 'version6.main',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('version', type=int, choices=sorted(ENTRY_POINTS))
    args, rest = parser.parse_known_args()
    module = importlib.import_module(ENTRY_POINTS[args.version])
    if args.version >= 5:
        module.main(rest)
    else:
        module.main()
