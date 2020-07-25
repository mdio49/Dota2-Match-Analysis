################################################################
# A program that uses a neural network to make predictions.
################################################################

import argparse, glob, gzip, json, math, os, torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from d2pred import *
from matches import *
from model import *

COLUMNS = ['id','seq_num','start_time','gm','region','r1','r2','r3','r4','r5','d1','d2','d3','d4','d5','duration','winner']
HEROES_INDEX = 5
DURATION_INDEX = 15
WINNER_INDEX = 16

# A neural network defined from a PyTorch data file.
class NeuralNetwork(Model):
    def __init__(self, path):
        self.net = torch.load(path)
    
    # Makes a prediction given the match data.
    def predict(self, match):
        # TODO
        pass

# Trains the network with the given arguments.
def train(args):
    # Set up the network.
    print("Setting up network.")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = None
    if args.net == 'linear':
        net = D2PredLinear().to(device)
    elif args.net == 'full':
        net = D2PredFull(args.hid).to(device)
    elif args.net == 'short':
        net = D2PredShort(args.hid).to(device)
    else:
        print("Invalid model specified.")
        exit()
    
    # Initialize weight values.
    if args.init_weights:
        net.init_weights(0, 1)

    # Initialize the optimizer.
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)

    # Collect the training and test data.
    X_train = None; Y_train = None; X_test = None; Y_test = None
    if args.use_cache:
        print("Retrieving data from cache.")
        X_train = torch.load("cache/X_train.pt")
        Y_train = torch.load("cache/Y_train.pt")
        X_test = torch.load("cache/X_test.pt")
        Y_test = torch.load("cache/Y_test.pt")
    else:
        # Collect the raw data from the CSV files.
        print("Collecting data.")
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")

        # Create the tensors.
        print("Creating tensors.")
        X_train, Y_train = create_tensors(train, args.batch_size)
        X_test, Y_test = create_tensors(test, 64)

        # Cache the data.
        if not args.no_cache:
            print("Caching tensors for future runs (run with --use_cache flag to use this data).")
            if not os.path.exists("cache"):
                os.mkdir("cache")
            torch.save(X_train, "cache/X_train.pt")
            torch.save(Y_train, "cache/Y_train.pt")
            torch.save(X_test, "cache/X_test.pt")
            torch.save(Y_test, "cache/Y_test.pt")
    
    # Print the initial performance for control purposes.
    accuracy, error, rmse = get_performance(net, X_test, Y_test, threshold=args.acc_thresh)
    print('Initial performance for control:')
    print('  accuracy = %7.4f%%' % accuracy)
    print('  error = %7.4f' % error)
    print('  RMS error: %7.4f' % rmse)

    # Iterate through the data for the given number of training epochs.
    print("Training network...")
    loss_sum = 0; loss_N = 0
    for epoch in range(0, args.epochs):
        # Train the network on the training data.
        for i in range(0, len(X_train)):
            # Pass the data into the network and perform backpropagation.
            data, target = X_train[i], Y_train[i]
            optimizer.zero_grad()
            output = net(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Keep a running total of the loss.
            loss_sum += loss.item()
            loss_N += 1

        # Compute the average loss.
        avg_loss = loss_sum / loss_N
        
        # Print the loss and performance at regular intervals.
        if (epoch + 1) % args.eval_interval == 0:
            accuracy, error, rmse = get_performance(net, X_test, Y_test, threshold=args.acc_thresh)
            print('ep%3d: loss = %7.4f' % (epoch + 1, loss.item()))
            print(('  accuracy = %7.4f%%' % accuracy) + (f" (with threshold = {args.acc_thresh})" if args.acc_thresh > 0 else ""))
            print('  error = %7.4f' % error)
            print('  RMS error: %7.4f' % rmse)
        
        # Stop training if the loss diminishes.
        if avg_loss < 0.00001:
            break
    
    torch.save(net, 'model.dat')
    print("Training complete. Network saved to 'model.dat'.")

# Creates the tensors 
def create_tensors(data, batch_size):
    # Convert the data into a numpy array, replacing radiant with 1 and dire with 0.
    data = data.replace({'winner': 'R'}, 1).replace({'winner': 'D'}, 0).to_numpy()

    # Create the input and output tensors for the given data.
    X = []; Y = []; k = 0
    while k < data.shape[0]:
        end = min(k + batch_size, data.shape[0])
        X_in = np.zeros((end - k, N_HEROES * 2))
        for i in range(0, end - k):
            row = data[k + i, HEROES_INDEX:(HEROES_INDEX + 10)]
            for j in row[0:5]:
                index = get_hero_index(j, radiant=True)
                X_in[i, index] = 1
            for j in row[5:10]:
                index = get_hero_index(j, radiant=False)
                X_in[i, index] = 1
        Y_in = data[k:end, WINNER_INDEX].astype(int)
        X.append(torch.from_numpy(X_in).float())
        Y.append(torch.from_numpy(Y_in).float())
        k += batch_size
    
    return X, Y

# Tests the network on the given test data and returns the number guessed correctly.
def get_performance(net, X_test, Y_test, threshold=0.0):
    n_correct = 0; n_total = 0; n_thresh = 0
    error_sum = 0
    rmse_sum = 0
    with torch.no_grad():
        for i in range(0, len(X_test)):
            data, target = X_test[i], Y_test[i]
            output = net(data)
            for j in range(0, len(output)):
                pred = output[j]
                correct = target[j]
                if (abs(pred - 0.5) >= threshold):
                    if (correct == 1 and pred >= 0.5) or (correct == 0 and pred <= 0.5):
                        n_correct += 1
                    n_thresh += 1

                error_sum += abs(pred - correct)
                rmse_sum += (pred - correct) * (pred - correct)
                n_total += 1
    
    accuracy = (n_correct / n_thresh) * 100
    error = error_sum / n_total
    rmse = math.sqrt(rmse_sum / n_total)
    return accuracy, error, rmse

# Updates the given data to a CSV file.
def update_csv(data, path):
    if os.path.exists(path):
        data.to_csv(path, mode="a", index=False, header=False)
    else:
        data.to_csv(path, mode="w", index=False, header=True)

# Processes JSON formatted match data and returns a list containing the columns
# for the data that would be placed in a CSV for the network to use.
def extract_match_data(match_data, api='steampowered'):
    # Parse the match data.
    match = Match(match_data, api=api)
    if match.match_id is None:
        return None
    
    # Get the heroes in the match.
    R, D = match.get_heroes()
    if not len(R) + len(D) == 10:
        return None

    # Format the list of data.
    record = [
        "?" if match.match_id is None else match.match_id,
        "?" if match.seq_num is None else match.seq_num,
        "?" if match.start_time is None else match.start_time,
        "?" if match.game_mode is None else match.game_mode,
        "?" if match.region is None else match.region,
        R[0], R[1], R[2], R[3], R[4],
        D[0], D[1], D[2], D[3], D[4],
        "?" if match.duration is None else match.duration,
        "R" if match.winner == Team.Radiant else "D" if match.winner == Team.Dire else "?"
    ]

    # Return the match data.
    return list(map(str, record))

# Cleans the data from the given data frame, extracting only relevant records.
def clean_data(data):
    # Filter only relevant game modes.
    data = data[data['gm'].isin([1, 2, 3, 5, 22])]

    # Filter out any matches with invalid hero IDs.
    data = data[
        (data['r1'] != 0) & (data['r2'] != 0) & (data['r3'] != 0) & (data['r4'] != 0) & (data['r5'] != 0) &
        (data['d1'] != 0) & (data['d2'] != 0) & (data['d3'] != 0) & (data['d4'] != 0) & (data['d5'] != 0)
    ]
    
    return data

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Dota 2 Match Prediction Tool")
    parser.add_argument("-process", metavar="DIR", type=str, help="Processes downloaded match data from the given directory and stores it in 'out.csv'.")
    parser.add_argument("--api", type=str, default='steampowered', help="The API that was used to obtain the data (default is 'steampowered').")
    parser.add_argument("-split", metavar="PATH", type=str, help="Takes a dataset and splits it randomly into training and test data.")
    parser.add_argument("--frac", type=float, default=0.9, help="The proportion of that dataset to take as training data.")
    parser.add_argument("-train", action="store_true", default=False, help="Trains the network using the files 'train.csv' and 'test.csv'.")
    parser.add_argument('--net', type=str, default='short', help='The network model to use for training.')
    parser.add_argument('--hid', type=int, default=100, help='The number of hidden nodes to use.')
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", metavar="SIZE", type=int, default=50, help="The maximum size of each batch.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--mom", type=float, default=0.2, help="Momentum.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--use_cache', action='store_true', default=False, help='Loads the tensors from cache as opposed to preprocessing the data from scratch.')
    parser.add_argument('--no_cache', action='store_true', default=False, help='Tensors will not be cached for the current run if this flag is used.')
    parser.add_argument('--eval_interval', metavar="N", type=int, default=1, help='The epoch interval at which to evaluate the network\'s performance while training.')
    parser.add_argument('--init_weights', action='store_true', default=False, help='Initializes weights to random values.')
    parser.add_argument('--acc_thresh', type=float, default=0.0, help='A threshold value from 0.5 to use when evaluating the accuracy of the network.')
    parser.add_argument("-predict", metavar="ID", type=int, help="Predicts the result for a given match ID.")
    parser.add_argument("-yasp_dump", metavar="PATH", type=str, help="Processes data from the YASP 3.5 Million Data Dump dataset. This dataset can be obtained from here: https://academictorrents.com/details/5c5deeb6cfe1c944044367d2e7465fd8bd2f4acf")
    args = parser.parse_args()

    # Process data to a CSV so it's usable by the machine learning model.
    if args.process:
        with open("out.csv", "w") as output:
            # Write the columns to the CSV.
            output.write(",".join(COLUMNS) + "\n")

            # Process each JSON file in the directory and append it to the CSV.
            dir_filter = os.path.join(args.process, args.api, "[0-9]*.json")
            for path in glob.glob(dir_filter, recursive=False):
                # Get the match data.
                match_data = {}
                with open(path, encoding='utf-8', mode="r") as json_file:
                    match_data = json.load(json_file)
                if 'error' in match_data:
                    continue
                    
                # Append the match data to the list of records.
                record = extract_match_data(match_data, api=args.api)
                if record is not None:
                    output.write(",".join(record) + "\n")

    # Split the given data into train and test sets.
    elif args.split:
        data = pd.read_csv(args.split)
        data = clean_data(data)
        data = data.sample(frac=1)
        
        n_rows = data.shape[0]
        split = int(n_rows * args.frac)

        train_data = data[0:split]
        test_data = data[split:n_rows]

        if not os.path.exists("data"):
            os.mkdir("data")

        train_data.to_csv("data/train.csv", index=False)
        test_data.to_csv("data/test.csv", index=False)
    
    # Train the network.
    elif args.train:
        train(args)

    # Make a prediction.
    elif args.predict:
        # Fetch the data for the match.
        keys = get_keys("keys.json")
        match = fetch_match(args.predict, keys=keys)

        # Parse the data for the given match.
        data = pd.DataFrame.from_records([extract_match_data(match)], columns=COLUMNS)
        
        # Pass the prediction into the model.
        net = torch.load("model.dat")
        X, Y = create_tensors(data, 1)
        output = net(X[0])

        # Print the actual result as well as the prediction.
        print("Winner: {}".format("Radiant" if data.loc[0,'winner'] == 'R' else "Dire" if data.loc[0,'winner'] == 'D' else "Unknown"))
        print("Prediction: {} ({})".format(get_normalised_output(output), "Radiant Adv." if output > 0.5 else "Dire Adv." if output < 0.5 else "Indeterminate"))
    
    # Process data from the YASP JSON dump dataset.
    elif args.yasp_dump:
        with open("yasp_dump.csv", "w") as output:
            output.write(",".join(COLUMNS) + "\n")
            completed = 0
            with gzip.open(args.yasp_dump, 'r') as file:
                for line in file:
                    # Checks if the first character is '{'.
                    if line[0] != 123:
                        continue
                    
                    # Each line is a JSON object that contains data for a single match.
                    match_data = json.loads(line)
                    record = extract_match_data(match_data, api=args.api)
                    if record is not None:
                        output.write(",".join(record) + "\n")

                    # Print progress every 1000 matches.
                    completed += 1
                    if (completed % 1000) == 0:
                        print("Completed: " + str(completed))
                        output.flush()

if __name__ == '__main__':
    main()
