import json


def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    load_json("\test.json")