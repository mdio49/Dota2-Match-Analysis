# Dota 2 Match Analysis and Prediction Tool
A tool that can be used to analyse and predict outcomes of matches in Dota 2.

## Programs
This tool has various script files that serve different purposes.
- bayesian.py: Used to train and test the Bayesian model.
- constants.py: Contains constants used to convert some data from the matches into a sensible format.
- counters.py: Analyses match data to produce a list of which heroes are better or worse against a particular hero.
- main.py: The main program for interacting with all the interfaces in the project.
- matches.py: Contains classes and interfaces used to analyse the match data.
- model.py: Contains the base abstract class for a model in this project.
- neural.py - Used to train and test the Neural Network model.

Most of the programs take command-line arguments, which can be listed using the --help flag.

### bayesian.py
This program can be used to train and test the Naive Bayes classifier that this tool uses. On the first run of the program without any command-line arguments, this program will produce the look-up table containing all the relevant probabilities it requires to make predictions (using match data from 'data/matches.csv'). From there, the model can be tested on the matches in 'data/test.csv' using the -test flag, and a specific match can be predicted using the -id flag followed by the ID of the match.

### counters.py
This program takes input a hero, and by default will list the top 10 heroes that this hero is best and worst against. Here is an example usage for the hero "Earthshaker":
'''
python counters.py -list earthshaker
'''

The number of heroes listed may also be changed using the --entries flag, for example, the following command would display the top 5 heroes for both best and worst against instead of the top 10:
'''
python counters.py -list earthshaker --entries 5
'''

### main.py
This program is designed to provide a nice, friendly interface for interacting with all the other programs in this project. The program can simply be run directly with no command-line arguments:
'''
python main.py
'''

Once the program has loaded, a welcome message is displayed and the user is presented with a command-line interface. You may type 'help' to list the available commands, although the syntax is not provided. To get started, you must load a model into the program. This can be achieved using the 'load' command, followed by the model that you wish to load (must reference a valid model configuration file from the 'models' directory). At the current state, you may choose between 'bayesian' or 'neural' (which provides the FC model):
'''
load bayesian
'''

To make sure the model has loaded correctly, type 'info' and it should come up with "Model: BayesianModel". From there, the first thing we could check is the win rates of each hero. One may check this by typing the 'rates' command (note that this may take about a minute to compute).

Next, we should attempt to compose a draft of heroes on either team; this can be managed using the 'draft' command. To add a hero on a particular team, you can do the following:
'''
draft add [radiant/dire] [hero]
'''

This would add a particular hero onto a particular team. For example, the following would add Earthshaker to the Radiant team:
'''
draft add radiant earthshaker
'''

To check that the hero has been added, we can use the following command to list the heroes on either side:
'''
draft list
'''

We should observe that "Earthshaker" is on the Radiant team, while the Dire has no heroes. If you made a mistake, you can remove a hero from a particular team by running the following command:
'''
draft remove [radiant/dire] [hero]
'''

Once the team composition is complete for both sides, the prograrm can predict the outcome using the 'predict' command, and can suggest heroes for either team using the 'suggest' command, optionally followed by the team. When you are done with the draft, simply run the following command to clear the draft:
'''
draft reset
'''

When you are done with the program, simply press Ctrl-Z or type 'exit' in the command-line interface.

### matches.py
This program is used to fetch and analyse match data. It mainly serves as an interface for other script files, but it may be interacted with to automatically generate and fetch match data and store them as .json files in a particular directory ('matches' by default). For example; the following command would fetch the match data for 100 matches starting at the match ID 50000000:
'''
python matches.py -fetch --start_id 50000000 --N 100 -api opendota
'''

The program can also produce a table listing the result of a match using the -get flag followed by the ID of the match.
'''
python matches.py -get 50000000 -api opendota
'''

This program may require a valid API key in order to fetch the data. The API to use can be specified using the -api flag; the default is 'steampowered' which requires a key. The 'opendota' API may also be used, which doesn't require an API key, but has a request limit rate of 60 requests per minute without one. API keys can be provided in the keys.json file.

### neural.py
This program can be used to train and test the Neural Network model that this tool uses. There are many command-line arguments that can be adjusted (which can be listed with the --help flag), though like bayesian.py, the model can be trained using the -train flag and tested using the -test flag. You can also predict a specific match using the -predict flag followed by the ID of the match.
