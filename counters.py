import argparse
import pandas as pd
from matches import *

def get_rating_str(rating):
    rating = rating * 100
    return ("+%.2f%%" % rating) if rating > 0 else ("%.2f%%" % rating)

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Dota 2 Counter-Picker")
    parser.add_argument("-list", metavar="HERO", type=str, help="Lists heroes that the given hero is best and worst against.")
    parser.add_argument("--entries", type=int, default=10, help="The number of entries for best and worst against to display.")
    parser.add_argument("-patch", type=int, help="Filters out matches with the given patch ID.")
    args = parser.parse_args()

    # Load the data frame,
    df = pd.read_csv("data/matches.csv")
    if args.patch:
        df = df[df['patch'] == args.patch]

    if args.list:
        # Get the ID of the hero inputted.
        h = get_hero_by_name(args.list)
        if h == 0:
            print("Invalid hero name.")
            exit()
        
        # Filter out the matches which contain the hero.
        matches_R = df[(df['r1'] == h) | (df['r2'] == h) | (df['r3'] == h) | (df['r4'] == h) | (df['r5'] == h)]
        matches_D = df[(df['d1'] == h) | (df['d2'] == h) | (df['d3'] == h) | (df['d4'] == h) | (df['d5'] == h)]

        # Enumerate through each hero to generate a list of relative ratings.
        ratings = []
        for hero_id, hero in HEROES.items():
            if hero_id == h:
                continue
            
            # Get all matches where the two heroes are against each other.
            mR = matches_R[
                (matches_R['d1'] == hero_id) |
                (matches_R['d2'] == hero_id) |
                (matches_R['d3'] == hero_id) |
                (matches_R['d4'] == hero_id) |
                (matches_R['d5'] == hero_id)
            ]
            mD = matches_D[
                (matches_D['r1'] == hero_id) |
                (matches_D['r2'] == hero_id) |
                (matches_D['r3'] == hero_id) |
                (matches_D['r4'] == hero_id) |
                (matches_D['r5'] == hero_id)
            ]

            # If there is no match data for the 2 heroes against each other, then skip it.
            if mR.empty and mD.empty:
                continue

            # Get all matches where the hero won.
            won = pd.concat([mR[mR['winner'] == 'R'], mD[mD['winner'] == 'D']])

            # Get the number of matches that each hero won.
            N = len(mR.index) + len(mD.index)
            hA = len(won.index)
            hB = N - hA
            
            # Get the relative rating and store it in the list.
            rating = (hA - hB) / (hA + hB)
            ratings.append((hero[0], rating))
        
        # Sort the ratings and then print the results.
        ratings.sort(key=lambda x: x[1], reverse=True)

        print("== Best Against: ==")
        for rating in ratings[:args.entries]:
            print("{0:20}\t{1}".format(rating[0], get_rating_str(rating[1])))
        
        print("\n== Worst Against: ==")
        for rating in ratings[-args.entries:]:
            print("{0:20}\t{1}".format(rating[0], get_rating_str(rating[1])))
    
if __name__ == '__main__':
    main()
