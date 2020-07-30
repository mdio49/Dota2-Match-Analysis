import argparse, torch
import numpy as np
import pandas as pd
from neural import *
from matches import *

def list_ratings(radiant):
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
    parser.add_argument("-winrates", action='store_true', default=False, help="Lists raw winrates for each hero.")
    args = parser.parse_args()

    if args.list_radiant:
        list_ratings(True)
    elif args.list_dire:
        list_ratings(False)
    elif args.winrates:
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

    elif args.predict:
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
