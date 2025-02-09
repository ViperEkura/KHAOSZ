import json
import warnings
from module import Khaosz, Transformer, Config


def generate(json_path: str):
    json_data = json.load(open(json_path, "r"))
    for data_index, data in enumerate(json_data):
        print(f"{data_index + 1}/{len(json_data)}")
        break


if __name__ == "__main__":
    generate("data/test.json")