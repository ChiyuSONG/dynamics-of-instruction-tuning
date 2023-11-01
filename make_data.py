import os
from pathlib import Path
import json
import jsonlines
from argparse import ArgumentParser
import random


def get_data_by_sz(data_path, data_size, dir_name):
    path = Path(data_path)
    data_files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    for data_file in data_files:
        data = []
        with open(data_file, "r", encoding="utf8") as f:
            for line in f:
                data.append(json.loads(line))

        file_name = "_".join(os.path.split(data_file)[-1].split("_")[:-1])

        if isinstance(data_size, int):
            assert len(data) >= data_size
            selected_data = random.sample(data, k=data_size)
        elif isinstance(data_size, dict):
            if file_name in data_size:
                sz = data_size[file_name]
            elif data_size["others"] == "uniform":
                sz = 1305   # uniformly replenish data from other abilities
                if file_name=="biology":
                    sz = None   # use all biology data as its full size may not be sufficient
                elif file_name=="chinese":
                    sz += 1 # add the reminder to keep the total quantity == 10k.
            elif data_size["others"] == "max":
                sz = None # use all available data for other abilities
            selected_data = data[:sz]

        file_name = "_".join(os.path.split(data_file)[-1].split("_")[:-1])
        with jsonlines.open(dir_name + "/" + file_name + ".json", "w") as writer:
            for sample in selected_data:
                writer.write(sample)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_type", type=str, default=None,
        choices=["curated-10", "curated-40", "curated-160", "curated-640", "curated-2560", "curated-10000",
                 "synthetic-10", "synthetic-40", "synthetic-160", "synthetic-640", "synthetic-2560", "synthetic-10000",
                 "synthetic-40960", "baseline", "reconstruct", "maximum", "mix-0", "mix-2560", "mix-40960"]
    )
    args = parser.parse_args()

    random.seed(0)
    dir_name = "data/" + args.data_type
    os.makedirs(dir_name, exist_ok=True)

    if args.data_type == "baseline":
        args.data_type = "curated-10000"
        print("Type 'baseline' is equivalent to 'curated-10000'.")

    if args.data_type == "mix-0":
        args.data_type = "maximum"
        print("Type 'mix-0' is equivalent to 'maximum'.")

    if "curated" in args.data_type:
        data_path = "data/curated/1000"
        data_size = int(args.data_type.split("-")[-1]) // 10
        get_data_by_sz(data_path, data_size, dir_name)
    elif "synthetic" in args.data_type:
        data_path = "data/synthetic/40960"
        data_size = int(args.data_type.split("-")[-1])
        get_data_by_sz(data_path, data_size, dir_name)
    elif args.data_type == "reconstruct":
        data_path = "data/curated/full"
        data_size = {"ethics":64, "role_play":64, "creative_writing":1000, "others": "uniform"}
        get_data_by_sz(data_path, data_size, dir_name)
    elif args.data_type == "maximum":
        data_path = "data/curated/full"
        data_size = {"ethics":64, "role_play":64, "creative_writing":1000, "others": "max"}
        get_data_by_sz(data_path, data_size, dir_name)
    elif "mix" in args.data_type:
        # mix two data sources
        data_path = "data/curated/full"
        data_size = {"ethics": 64, "role_play": 64, "creative_writing": 1000, "others": "max"}
        get_data_by_sz(data_path, data_size, dir_name)

        data_path = "data/synthetic/40960"
        data_size = int(args.data_type.split("-")[-1])
        get_data_by_sz(data_path, data_size, dir_name)

    print("Make Data Done!")

if __name__ == "__main__":
    main()
