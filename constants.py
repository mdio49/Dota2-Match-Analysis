import json

def cluster(index):
    with open("constants/cluster.json", "r") as file:
        data = json.load(file)
        key = str(index)
        return None if key not in data.keys() else data[key]
