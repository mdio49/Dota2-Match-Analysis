import argparse, torch
import numpy as np
from d2pred import *
from matches import *

def list_winrates(radiant):
    print("-- Individual Hero Ratings --")
    ratings = []
    net = torch.load("model.dat")
    zeros = np.zeros((1, N_HEROES * 2))
    with torch.no_grad():
        for hero_id, hero in HEROES.items():
            index = get_hero_index(hero_id, radiant)
            X = torch.from_numpy(zeros).float()
            X[0, index] = 1
            output = net(X)
            ratings.append((hero[0], output))
    
    ratings.sort(key=lambda x: x[1], reverse=True)
    for rating in ratings:
        print("{0:20}\t{1}".format(rating[0], get_normalised_output(rating[1])))

def get_team():
    team = []
    while len(team) < 5:
        hero_name = input().strip()
        if hero_name == '\x04':
            break
        hero_id = get_hero_by_name(hero_name)
        if hero_id > 0:
            team.append(hero_id)
        else:
            print("Invalid hero name.")
    return team

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Dota 2 Hero Selector")
    parser.add_argument("-predict", action='store_true', default=False, help="Make a prediciton given a list of heroes.")
    parser.add_argument("-list_radiant", action='store_true', default=False, help="Lists ratings for heroes on the radiant side.")
    parser.add_argument("-list_dire", action='store_true', default=False, help="Lists ratings for heroes on the dire side.")
    args = parser.parse_args()

    if args.list_radiant:
        list_winrates(True)
    elif args.list_dire:
        list_winrates(False)
    
    if args.predict:
        # Ask the user for some heroes.
        print("Radiant Heroes:")
        radiant_heroes = get_team()

        print("Dire Heroes:")
        dire_heroes = get_team()

        # Make a prediction.
        net = torch.load("model.dat")
        X = torch.zeros((1, N_HEROES * 2)).float()
        for hero_id in radiant_heroes:
            index = get_hero_index(hero_id, radiant=True)
            X[0, index] = 1
        for hero_id in dire_heroes:
            index = get_hero_index(hero_id, radiant=False)
            X[0, index] = 1
        output = net(X)

        print("Prediction: {} ({})".format(get_normalised_output(output), "Radiant Adv." if output > 0.5 else "Dire Adv." if output < 0.5 else "Indeterminate"))

if __name__ == '__main__':
    main()
