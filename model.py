from abc import ABC, abstractmethod

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

# Prints the results of a prediction given the winner, prediction and advantage.
def print_prediction(winner, pred, adv):
    print("Winner: " + str(winner))
    print(f"Prediction: {pred} Advantage (+{'%.2f' % (adv * 100)}%)")
    