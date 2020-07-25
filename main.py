################################################################
# The main program that interacts with all the other interfaces.
################################################################

import os, json
from matches import *
from model import *
from bayesian import BayesianModel
from neural import NeuralNetwork

# The list of available commands.
cmd_list = []

# A class for a command.
class Command:
    def __init__(self, name, description, execute):
        self.name = name
        self.description = description
        self.execute = execute

# Loads the specified model into the program.
def load_model(config, model_id):
    path = f"models/{model_id}.json"
    if not os.path.exists(path):
        print(f"Model configuration file '{path}' does not exist.")
        return
    
    model = None; model_name = None
    with open(path, "r") as file:
        model_config = json.load(file)
        model_name = model_config['name'] if 'name' in model_config.keys() else model_id
        model_type = model_config['type'] if 'type' in model_config.keys() else None
        data = model_config['data'] if 'data' in model_config.keys() else {}
        if model_type == "bayesian":
            table_path = open(data['table'], "r")
            table = json.load(table_path)
            model = BayesianModel(table)
        elif model_type == "neural":
            model = NeuralNetwork(data['path'])
        else:
            print("Invalid model type defined in configuration file.")
            return
    
    if model is not None:
        config['model'] = model
        print(f"Model '{model_name}' loaded successfully.")
    
    return config

def list_commands():
    for cmd in cmd_list:
        print("{0:20} {1}".format(cmd.name, cmd.description))

def info(config):
    model = config['model']
    print(f"Model: {type(model).__name__}")

def process_input(cmd_list, user_input):
    split = user_input.strip().split(' ')
    name = split[0]
    args = split[1:]
    cmd = next((x for x in cmd_list if x.name == name), None)
    return cmd, args

def predict(config, match_id):
    model = config['model']
    if model is None:
        model_not_loaded()
        return
    
    # Fetch the match data.
    match_data = fetch_match(match_id, keys=config['keys'])
    if 'error' in match_data.keys():
        print(f"Couldn't load match data: {match_data['error']}")
        return

    # Parse the match data.
    match = Match(match_data)

    # Make the prediction.
    pred, rating = model.predict(match)
    print_prediction(match.winner, pred, rating)

def model_not_loaded():
    print("Model does not exist. Please load a model into the program using the 'load' command.")

def main():
    print("Welcome to the Dota 2 Match prediction tool. Type 'help' for a list of available commands.")

    # Produce the list of available commands.
    cmd_list.append(Command("help", "Displays a list of all available commands.", lambda config, args: list_commands()))
    cmd_list.append(Command("info", "Displays the current state of the program.", lambda config, args: info(config)))
    cmd_list.append(Command("load", "Loads the given model into the program.", lambda config, args: load_model(config, args[0])))
    cmd_list.append(Command("predict", "Predicts the outcome for the given match ID.", lambda config, args: predict(config, args[0])))
    cmd_list.append(Command("exit", "Exits the program.", lambda config, args: exit()))
    
    # Set up the configuration parameters.
    config = {
        'model': None,
        'keys': {}
    }

    # Load the API keys.
    keys_path = "keys.json"
    if os.path.exists(keys_path):
        config['keys'] = get_keys(keys_path)

    while True:
        print("> ", end='')
        try:
            user_input = input()
            cmd, args = process_input(cmd_list, user_input)
            if cmd is None:
                print("Unknown command. Type 'help' for a list of available commands.")
                continue
            
            try:
                new_config = cmd.execute(config, args)
                if new_config is not None:
                    config = new_config   
            except IndexError:
                print("Invalid arguments specified.")
            
        except EOFError:
            break

if __name__ == '__main__':
    main()
