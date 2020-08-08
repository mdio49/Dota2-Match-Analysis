from abc import ABC, abstractmethod
from matches import Team

# An abstract class for a model.
class Model(ABC):
    def __init__(self):
        pass
    
    # Predicts the outcome of the given match, returning a tuple containing the
    # team it predicts to win, and a decimal between 0 to 1 indicating the
    # relative advantage the team has (i.e. how confident the prediction is).
    @abstractmethod
    def predict(self, match):
        pass
    
    # Similar to 'predict', but makes a prediction given a list of heroes on
    # either side (the lists may be incomplete).
    @abstractmethod
    def predict_heroes(self, R, D):
        pass

    # Tets the model on the given set of test data.
    def test(self, data, thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        Nt = len(thresholds)

        # Iterate through each match in the test dataset.
        n_correct = [0] * Nt; n_total = [0] * Nt
        for index, match in data.iterrows():
            # Get the heroes in the match.
            R = [match['r1'], match['r2'], match['r3'], match['r4'], match['r5']]
            D = [match['d1'], match['d2'], match['d3'], match['d4'], match['d5']]

            # Get the actual winner.
            target = Team.Radiant if match['winner'] == 'R' else Team.Dire if match['winner'] == 'D' else None
            if target is None:
                continue

            # Make a prediction.
            pred, rating = self.predict_heroes(R, D)
            for i in range(0, Nt):
                if rating >= thresholds[i]:
                    # Test if the prediction matches the target.
                    if target == pred:
                        n_correct[i] += 1
                
                    n_total[i] += 1
            
            if (index + 1) % 1000 == 0:
                print("Completed: " + str(index + 1))
        
        # Format the accuracy strings.
        accuracy_str = [None] * Nt
        for i in range(0, Nt):
            if n_correct[i] == 0:
                accuracy_str[i] = "--%"
            elif n_correct[i] == n_total[i]:
                accuracy_str[i] = "100%"
            else:
                accuracy_str[i] = "%.2f%%" % ((n_correct[i] / n_total[i]) * 100)
        
        # Generate the string formatting mask.
        mask = "{0:10}"
        for i in range(0, Nt):
            mask += " | {" + str(i + 1) + ":6}"
        
        # Convert the numeric lists into lists of strings.
        thresholds = list(map(str, thresholds))
        n_correct = list(map(str, n_correct))
        n_total = list(map(str, n_total))

        # Print the table of performance results.
        print("Testing complete. Performance results:")
        print(mask.format("Threshold", *thresholds))
        print(mask.format("Accuracy", *accuracy_str))
        print(mask.format("Correct", *n_correct))
        print(mask.format("Total", *n_total))

# Prints the results of a prediction given the winner, prediction and advantage.
def print_prediction(winner, pred, adv):
    print("Winner: " + str(winner))
    print(f"Prediction: {pred} Advantage (+{'%.2f' % (adv * 100)}%)")
    