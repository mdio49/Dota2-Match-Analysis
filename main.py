################################################################
# The main program that interacts with all the other interfaces.
################################################################

import os, json
import pandas as pd
from abc import ABC, abstractmethod
from matches import *
from model import *
from bayesian import *
from neural import *

# The list of available commands.
cmd_list = []

# An abstract class for a command.
class Command(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, config, args):
        pass

class LambdaCommand(Command):
    def __init__(self, name, description, exec_lambda):
        self.name = name
        self.description = description
        self.exec_lambda = exec_lambda
        
    def execute(self, config, args):
        self.exec_lambda(config, args)

class Cmd_Draft(Command):
    def __init__(self):
        super().__init__("draft", "Manages the team composition on either side.")
    
    def execute(self, config, args):
        subcmd = args[0]
        if subcmd == "add":
            config = self.draft_add(config, ' '.join(args[2:]), args[1])
        elif subcmd == "remove":
            config = self.draft_remove(config, ' '.join(args[2:]), args[1])
        elif subcmd == "list":
            config = self.draft_list(config)
        elif subcmd == "reset":
            config = self.draft_reset(config)
        else:
            print("Invalid sub-command. Expected 'add', 'remove', 'list' or 'reset'.")
        return config

    def draft_add(self, config, name, team):
        hero_id = get_hero_by_name(name)
        if hero_id > 0:
            full_name = HEROES[hero_id][0]
            if hero_id in config['radiant_heroes'] or hero_id in config['dire_heroes']:
                print(f'{full_name} is has already been selected!')
            else:
                if team.lower() == 'radiant':
                    if len(config['radiant_heroes']) < 5:
                        config['radiant_heroes'].append(hero_id)
                        print(f'{full_name} has been added to the Radiant team!')
                    else:
                        print("Radiant already has 5 heroes!")
                elif team.lower() == 'dire':
                    if len(config['dire_heroes']) < 5:
                        config['dire_heroes'].append(hero_id)
                        print(f'{full_name} has been added to the Dire team!')
                    else:
                        print("Dire already has 5 heroes!")
                else:
                    print("Invalid team name!")
        else:
            print("Invalid hero name!")
        return config

    def draft_remove(self, config, name, team):
        hero_id = get_hero_by_name(name)
        if hero_id > 0:
            full_name = HEROES[hero_id][0]
            if team.lower() == 'radiant':
                if hero_id in config['radiant_heroes']:
                    config['radiant_heroes'].remove(hero_id)
                    print(f'{full_name} has been removed from the Radiant team!')
                else:
                    print(f'{full_name} is not part of the Radiant team!')
            elif team.lower() == 'dire':
                if hero_id in config['dire_heroes']:
                    config['dire_heroes'].remove(hero_id)
                    print(f'{full_name} has been removed from the Dire team!')
                else:
                    print(f'{full_name} is not part of the Dire team!')
            else:
                print("Invalid team name!")
        else:
            print("Invalid hero name!")
        return config

    def draft_list(self, config):
        print("== THE RADIANT ==")
        if len(config['radiant_heroes']) == 0:
            print("Nothing to display.")
        for hero_id in config['radiant_heroes']:
            print(HEROES[hero_id][0])
        print("\n== THE DIRE ==")
        if len(config['dire_heroes']) == 0:
            print("Nothing to display.")
        for hero_id in config['dire_heroes']:
            print(HEROES[hero_id][0])
        return config

    def draft_reset(self, config):
        config['radiant_heroes'].clear()
        config['dire_heroes'].clear()
        print("All heroes have been removed from both teams!")
        return config

class Cmd_Suggest(Command):
    def __init__(self):
        super().__init__("suggest", "Makes hero suggestions given the current draft.")
    
    def execute(self, config, args):
        model = config['model']
        if model is None:
            model_not_loaded()
        else:
            # Get the current hero draft and predicted advantage.
            R, D = config['radiant_heroes'], config['dire_heroes']
            pred, rating = model.predict_heroes(R, D)

            # Make suggestions for each team where applicable.
            if len(args) == 0 or args[0] == 'radiant':
                print("Radiant suggestions:")
                rating = -rating if pred == Team.Dire else rating
                self.suggest_team(model, rating, R, D, Team.Radiant)
                print()
            if len(args) == 0 or args[0] == 'dire':
                print("Dire suggestions:")
                rating = -rating if pred == Team.Radiant else rating
                self.suggest_team(model, rating, R, D, Team.Dire)
                print()
                
        return config
    
    def suggest_team(self, model, adv, R, D, team, N=10):
        if team == Team.Radiant and len(R) >= 5:
            print("Radiant team is full!")
            return
        elif team == Team.Dire and len(D) >= 5:
            print("Dire team is full!")
            return
        
        # Enumerate through each hero.
        ratings = []
        for hero_id, names in HEROES.items():
            # If the hero is already in the game, then skip it.
            if hero_id in R or hero_id in D:
                continue
            
            # Add the hero to the correct team.
            newR = R.copy(); newD = D.copy()
            if team == Team.Radiant:
                newR.append(hero_id)
            elif team == Team.Dire:
                newD.append(hero_id)

            # Make a prediction given the new hypothetical team composition.
            new_adv = 0
            try:
                pred, new_adv = model.predict_heroes(newR, newD)
                new_adv *= -1 if pred != team else 1
            except ArithmeticError:
                continue

            # Calculate the relative advantage that adding this hero has on the rating.
            rating = new_adv - adv

            # Append the rating to the list.
            ratings.append((names[0], rating))
        
        # Sort the heroes by their rating, and then print the top N heroes.
        ratings.sort(key=lambda x: x[1], reverse=True)
        for rating in ratings[0:N]:
            print("{0:20}\t{1}".format(rating[0], f"{'+' if rating[1] >= 0 else ''}{'%.2f' % (rating[1] * 100)}%"))

class Cmd_Predict(Command):
    def __init__(self):
        super().__init__("predict", "Makes a prediction given the current draft of heroes.")
    
    def execute(self, config, args):
        model = config['model']
        if model is None:
            model_not_loaded()
        else:
            pred, rating = model.predict_heroes(config['radiant_heroes'], config['dire_heroes'])
            print(f"Prediction: {pred} Advantage (+{'%.2f' % (rating * 100)}%)")
        return config

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

def win_rates():
    df = pd.read_csv("data/matches.csv")
    winrates = []
    for h, names in HEROES.items():
        # Get all matches which the hero participated in.
        matches_R = df[(df['r1'] == h) | (df['r2'] == h) | (df['r3'] == h) | (df['r4'] == h) | (df['r5'] == h)]
        matches_D = df[(df['d1'] == h) | (df['d2'] == h) | (df['d3'] == h) | (df['d4'] == h) | (df['d5'] == h)]
        matches = pd.concat([matches_R, matches_D])
        
        # Of those matches, find the matches where the hero won.
        won = pd.concat([matches_R[matches_R['winner'] == "R"], matches_D[matches_D['winner'] == "D"]])
        
        if not matches.empty:
            # Calculate the win rate.
            win_rate = len(won.index) / len(matches.index)

            # Store the win rate in the list.
            winrates.append((names[0], win_rate))
    
    # Sort the win rates in ascending order and display them to the screen.
    winrates.sort(key=lambda x: x[1], reverse=True)
    for hero, win_rate in winrates:
        print("{0:20}\t{1}".format(hero, ("%.2f%%" % (win_rate * 100))))

def model_not_loaded():
    print("Model does not exist. Please load a model into the program using the 'load' command.")

def main():
    print("Welcome to the Dota 2 Match prediction tool. Type 'help' for a list of available commands.")

    # Produce the list of available commands.
    cmd_list.append(LambdaCommand("help", "Displays a list of all available commands.", lambda config, args: list_commands()))
    cmd_list.append(LambdaCommand("info", "Displays the current state of the program.", lambda config, args: info(config)))
    cmd_list.append(LambdaCommand("load", "Loads the given model into the program.", lambda config, args: load_model(config, args[0])))
    #cmd_list.append(LambdaCommand("predict", "Predicts the outcome for the given match ID.", lambda config, args: predict(config, args[0])))
    cmd_list.append(Cmd_Draft())
    cmd_list.append(Cmd_Suggest())
    cmd_list.append(Cmd_Predict())
    cmd_list.append(LambdaCommand("rates", "Lists the winrates for each hero (may take a while to compute).", lambda config, args: win_rates()))
    cmd_list.append(LambdaCommand("exit", "Exits the program.", lambda config, args: exit()))
    
    # Set up the configuration parameters.
    config = {
        'model': None,
        'keys': {},
        'radiant_heroes': [],
        'dire_heroes': []
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
