from argparse import  ArgumentParser
from pathlib import Path
import shutil
import subprocess


def do_segmentation(in_pc: Path, out_root: Path) -> None:
    out_root.mkdir(exist_ok=True, parents=True)
    shutil.copy(in_pc, out_root / f'gt{in_pc.suffix}')
    for algo in ['li', 'dalponte', 'silva', 'watershed']:
        out_path = out_root / f'{algo}.laz'
        script = f'source("{Path(__file__).parent / "segment_trees.R"}"); segment_and_save("{in_pc}", "{out_path}", "{algo}");'
        command = ['Rscript', '-e', script]
        subprocess.run(command)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('in_root', help='Root directory containing all target point clouds')
    parser.add_argument('out_root', help='Root directory to write outputs')
    args = parser.parse_args()

    in_root = Path(args.in_root).resolve()
    in_pcs = list(in_root.rglob('*.las')) + list(in_root.rglob('*.laz'))
    out_root = Path(args.out_root).resolve()

    for pc in in_pcs:
        out_path = out_root / pc.relative_to(in_root).parent / pc.stem
        do_segmentation(pc, out_path)
