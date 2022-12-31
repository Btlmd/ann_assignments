import random
from argparse import ArgumentParser, Namespace
import re
from pathlib import Path

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--choice_output_dir", default="choice", type=str)
    parser.add_argument("--reg_exp", default=r".*\.txt", type=str)
    parser.add_argument("--range", nargs=2, default=[8, 5008], type=int)
    parser.add_argument("--num_choice", default=10, type=int)
    parser.add_argument("--seed", default=2022, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    pattern = re.compile(args.reg_exp)
    output = Path(args.choice_output_dir)
    output.mkdir(exist_ok=True)

    for file in Path(args.output_dir).iterdir():
        if pattern.match(file.name):
            print("FOUND", file.name)
            with open(file, "r") as f:
                lines = f.read().strip().split("\n")
            cond, lines = lines[: args.range[0]], lines[slice(*args.range)]
            rng = random.Random(args.seed)
            rng.shuffle(lines)
            choices = lines[: args.num_choice]
            with open(output / file.name, "w") as f:
                for line in cond + choices:
                    print(line, file=f)
