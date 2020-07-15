################################################################
# A program that uses a Bayesian classifier to make predictions.
################################################################

import argparse, json, os
import pandas as pd
from matches import *

# Gets the matches with have a hero on a particular team.
def get_matches_with_hero_on_team(data, hero, team):
    matches = None
    if team == Team.Radiant:
        matches = data[(data['r1'] == hero) | (data['r2'] == hero) | (data['r3'] == hero) | (data['r4'] == hero) | (data['r5'] == hero)]
    elif team == Team.Dire:
        matches = data[(data['d1'] == hero) | (data['d2'] == hero) | (data['d3'] == hero) | (data['d4'] == hero) | (data['d5'] == hero)]
    return matches

# Calculates the probability that a hero is on a given team.
def pick_rate(data, hero, team, winner=None):
    total = data
    matches = get_matches_with_hero_on_team(data, hero, team)
    if winner == Team.Radiant:
        matches = matches[matches['winner'] == 'R']
        total = data[data['winner'] == 'R']
    elif winner == Team.Dire:
        matches = matches[matches['winner'] == 'D']
        total = data[data['winner'] == 'D']
    return len(matches.index) / len(total.index)

# Trains the network on the given data to produce a probability table.
def train(data):
    P = {}

    # Calculate overall win rates for each team.
    P['W = Radiant'] = len(data[data['winner'] == 'R'].index) / len(data.index)
    P['W = Dire'] = len(data[data['winner'] == 'D'].index) / len(data.index)

    # Calculate the pick rates for each hero.
    for team in [Team.Radiant, Team.Dire]:
        for hero_id, names in HEROES.items():
            P[f'{hero_id} in {team}'] = pick_rate(data, hero_id, team)
        
    # Calculate the pick rates for each hero given a winner.
    for winner in [Team.Radiant, Team.Dire]:
        for team in [Team.Radiant, Team.Dire]:
            for hero_id, names in HEROES.items():
                P[f'{hero_id} in {team} | W = {winner}'] = pick_rate(data, hero_id, team, winner=winner)
    
    # Return the probability table.
    return P

# Predicts the outcome of the match with the given match ID.
def predict(P, match_id):
    # Fetch the match data.
    keys = get_keys("keys.json")
    match_data = fetch_match(match_id, keys=keys)
    if 'error' in match_data.keys():
        print(f"Couldn't load match data: {match_data['error']}")
        exit()

    # Parse the match data.
    match = Match(match_data)
    R, D = match.get_heroes()

    # Make the prediction.
    pred, rating = predict_heroes(P, R, D)
    print("Winner: " + str(match.winner))
    print(f"Prediction: {pred} Advantage (+{'%.2f' % (rating * 100)}%)")

# Makes a prediction given the heroes on both teams. Returns the team that it
# predicts will win as well as the relative advantage that the team has.
def predict_heroes(P, radiant_heroes, dire_heroes):
    # Let x_i be the heroes in Radiant and y_i be the heroes in Dire.
    # Let T be the event that each x_i is in Radiant and each y_i is in Dire.
    # We wish to calculate P(W = R | T) = P(T | W = R) * P(W = R) / P(T) and
    # the same for P(W = D | T) using Bayes' Theorem.

    # Calculate the probability for the team composition (assuming independence).
    P['T'] = 1.0
    for hero in radiant_heroes:
        P['T'] *= P[f'{hero} in Radiant']
    for hero in dire_heroes:
        P['T'] *= P[f'{hero} in Dire']

    # Calculate the probability for the team composition given that a particular team won.
    P['T | W=R'] = 1.0
    for hero in radiant_heroes:
        P['T | W=R'] *= P[f'{hero} in Radiant | W = Radiant']
    for hero in dire_heroes:
        P['T | W=R'] *= P[f'{hero} in Dire | W = Radiant']

    P['T | W=D'] = 1.0
    for hero in radiant_heroes:
        P['T | W=D'] *= P[f'{hero} in Radiant | W = Dire']
    for hero in dire_heroes:
        P['T | W=D'] *= P[f'{hero} in Dire | W = Dire']

    # Calculate the probability of each team winning.
    P['W=R | T'] = (P['T | W=R'] * P['W = Radiant']) / P['T']
    P['W=D | T'] = (P['T | W=D'] * P['W = Dire']) / P['T']

    # Compute the effectiveness and print the prediction:
    if P['W=R | T'] >= P['W=D | T']:
        rating = (P['W=R | T'] - P['W=D | T']) / (P['W=R | T'] + P['W=D | T'])
        return Team.Radiant, rating
    else:
        rating = (P['W=D | T'] - P['W=R | T']) / (P['W=R | T'] + P['W=D | T'])
        return Team.Dire, rating

# Tets the model on the given set of test data.
def test(P, data, threshold=0.0):
    # Iterate through each match in the test dataset.
    n_correct = 0; n_total = 0
    for index, match in data.iterrows():
        # Get the heroes in the match.
        R = [match['r1'], match['r2'], match['r3'], match['r4'], match['r5']]
        D = [match['d1'], match['d2'], match['d3'], match['d4'], match['d5']]

        # Get the actual winner.
        target = Team.Radiant if match['winner'] == 'R' else Team.Dire if match['winner'] == 'D' else None
        if target is None:
            continue

        # Make a prediction.
        pred, rating = predict_heroes(P, R, D)
        if rating < threshold:
            continue
        
        # Test if the prediction matchesh the target.
        if target == pred:
            n_correct += 1
        
        n_total += 1
    
    threshold_str = (" (with threshold = %.2f)" % threshold) if threshold > 0 else ""
    print(("Testing complete. Accuracy = %.2f%% (%d / %d)" % ((n_correct / n_total) * 100, n_correct, n_total)) + threshold_str)

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Dota 2 Match Prediction Tool (Bayesian)")
    parser.add_argument('-id', type=int, help='The match ID to predict.')
    parser.add_argument('-test', action="store_true", default=False, help='Tests the model.')
    parser.add_argument('--threshold', type=float, default=0.0, help='The threshold rating to use when considering if the prediction is correct or not.')
    args = parser.parse_args()

    # Load or produce the look-up table.
    P = {}; cache_dir = "cache/bayesian_table.json"
    if not os.path.exists(cache_dir):
        print("Training model for first time use to produce look-up table.")
        data = pd.read_csv("data/matches.csv")
        P = train(data)
        with open(cache_dir, "w") as file:
            json.dump(P, file)
    else:
        print("Loading look-up table from cache.")
        with open(cache_dir, "r") as file:
            P = json.load(file)

    # Make the prediction.
    if args.test:
        print("Testing model.")
        test_data = pd.read_csv("data/test.csv")
        test(P, test_data, threshold=args.threshold)
    elif args.id:
        print("Making prediction.")
        predict(P, args.id)
    else:
        print("No command-line arguments specified.")
        parser.print_usage()

if __name__ == '__main__':
    main()
