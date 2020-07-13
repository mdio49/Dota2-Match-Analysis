################################################################
# Fetches data for Dota 2 matches using Python's request library.
# Also provides some interfaces to process the data and extract
# information from them so they may be used by another program.
################################################################

import argparse, os, requests, json
from enum import Enum

# The list of heroes with their corresponding ID and aliases.
HEROES = {1: ['Anti-Mage','Anti Mage','am'], 2: ['Axe'], 3: ['Bane'], 4: ['Bloodseeker','blood','bs'], 5: ['Crystal Maiden','crystal','cm'], 6: ['Drow Ranger','drow'], 7: ['Earthshaker','shaker'], 8: ['Juggernaut','jugg','jug'], 9: ['Mirana','potm'], 10: ['Morphling','morph'], 11: ['Shadow Fiend','sf'], 12: ['Phantom Lancer','pl'], 13: ['Puck'], 14: ['Pudge'], 15: ['Razor'], 16: ['Sand King','sk'], 17: ['Storm Spirit','storm'], 18: ['Sven'], 19: ['Tiny'], 20: ['Vengeful Spirit','venge'], 21: ['Windranger','wr'], 22: ['Zeus'], 23: ['Kunkka','kk'], 25: ['Lina'], 26: ['Lion'], 27: ['Shadow Shaman','rhasta','rasta'], 28: ['Slardar'], 29: ['Tidehunter','tide'], 30: ['Witch Doctor','wd'], 31: ['Lich'], 32: ['Riki'], 33: ['Enigma'], 34: ['Tinker'], 35: ['Sniper'], 36: ['Necrophos','necro'], 37: ['Warlock'], 38: ['Beastmaster','beast'], 39: ['Queen of Pain','qop'], 40: ['Venomancer','veno'], 41: ['Faceless Void','faceless','fv'], 42: ['Wraith King','wraith','wk'], 43: ['Death Prophet','dp'], 44: ['Phantom Assassin','pa'], 45: ['Pugna'], 46: ['Templar Assassin','ta'], 47: ['Viper'], 48: ['Luna'], 49: ['Dragon Knight','dk'], 50: ['Dazzle'], 51: ['Clockwerk','clock','cw'], 52: ['Leshrac','lesh'], 53: ['Nature\'s Prophet','np'], 54: ['Lifestealer','naix','ls'], 55: ['Dark Seer','ds'], 56: ['Clinkz','bone'], 57: ['Omniknight','omni'], 58: ['Enchantress','ench'], 59: ['Huskar','husk'], 60: ['Night Stalker','nightstalker','ns'], 61: ['Broodmother','brood','bm','spider'], 62: ['Bounty Hunter','bounty','bh'], 63: ['Weaver'], 64: ['Jakiro','jak'], 65: ['Batrider','bat'], 66: ['Chen'], 67: ['Spectre','spec'], 68: ['Ancient Apparition','aa'], 69: ['Doom'], 70: ['Ursa','bear'], 71: ['Spirit Breaker','bara','sb'], 72: ['Gyrocopter','gyro'], 73: ['Alchemist','alch'], 74: ['Invoker','voker'], 75: ['Silencer'], 76: ['Outworld Devourer','od'], 77: ['Lycan'], 78: ['Brewmaster','brew','panda'], 79: ['Shadow Demon','sd'], 80: ['Lone Druid','ld'], 81: ['Chaos Knight','ck'], 82: ['Meepo'], 83: ['Treant Protector','treant','tree','tp'], 84: ['Ogre Magi','ogre','magi'], 85: ['Undying'], 86: ['Rubick'], 87: ['Disruptor'], 88: ['Nyx Assassin','nyx'], 89: ['Naga Siren','naga'], 90: ['Keeper of the Light','kotl'], 91: ['Io','wisp'], 92: ['Visage'], 93: ['Slark'], 94: ['Medusa','dusa'], 95: ['Troll Warlord','troll','tw'], 96: ['Centaur Warrunner','centaur','cent','cw'], 97: ['Magnus','mag'], 98: ['Timbersaw','timber'], 99: ['Bristleback','bristle','bb'], 100: ['Tusk'], 101: ['Skywrath Mage','skywrath','sky mage','skymage','sky','sm'], 102: ['Abaddon','abba'], 103: ['Elder Titan','et'], 104: ['Legion Commander','legion','lc'], 105: ['Techies'], 106: ['Ember Spirit','ember'], 107: ['Earth Spirit'], 108: ['Underlord','pitlord','pit'], 109: ['Terrorblade','terror','tb'], 110: ['Phoenix','bird'], 111: ['Oracle'], 112: ['Winter Wyvern','winter','ww'], 113: ['Arc Warden','arc','aw'], 114: ['Monkey King','monkey','mk'], 119: ['Dark Willow','willow','dw'], 120: ['Pangolier','pango'], 121: ['Grimstroke','grim'], 126: ['Void Spirit'], 128: ['Snapfire','snap'], 129: ['Mars']}

# The total number of heroes.
N_HEROES = len(HEROES.keys())

# The list of game-modes.
GAME_MODES = {
    0: None,
    1: 'All Pick',
    2: 'Captain\'s Mode',
    3: 'Random Draft',
    4: 'Single Draft',
    5: 'All Random',
    6: 'Intro',
    7: 'Diretide',
    8: 'Reverse Captain\'s Mode',
    9: 'The Greeviling',
    10: 'Tutorial',
    11: 'Mid Only',
    12: 'Least Played',
    13: 'New Player Pool',
    14: 'Compendium Matchmaking',
    15: 'Co-op vs Bots',
    16: 'Captain\'s Draft',
    18: 'Ability Draft',
    20: 'All Random Deathmatch',
    21: '1v1 Mid Only',
    22: 'Ranked Matchmaking',
    23: 'Turbo Mode'
}

# The list of skill levels.
SKILLS = {None: '?', 1: 'Normal Skill', 2: 'High Skill', 3: 'Very High Skill'}

# The two possible teams: Radiant and Dire.
class Team(Enum):
    Radiant = 1,
    Dire = 0

    def __str__(self):
        return self.name

# A class that processes match data.
class Match:
    def __init__(self, data, api='steampowered'):
        self.match_id = None if 'match_id' not in data.keys() else data['match_id']
        self.game_mode = None if 'game_mode' not in data.keys() else data['game_mode']
        self.duration = None if 'duration' not in data.keys() else (data['duration'] / 60.0)
        self.winner = None if 'radiant_win' not in data.keys() else Team.Radiant if data['radiant_win'] else Team.Dire

        self.players = []
        if 'players' in data.keys():
            for player in data['players']:
                self.players.append(Player(player, api=api))

        self.patch = None
        self.skill = None
        self.region = None

        if api == 'steampowered':
            # TODO
            pass
            
        if api == 'opendota':
            self.patch = None if 'patch' not in data.keys() else data['patch']
            self.skill = None if 'skill' not in data.keys() else data['skill']
            self.region = None if 'region' not in data.keys() else data['region']

    # Gets the heroes on each side of the match.
    def get_heroes(self):
        radiant_heroes = []; dire_heroes = []
        for player in self.players:
            if player.team == Team.Radiant:
                radiant_heroes.append(player.hero_id)
            elif player.team == Team.Dire:
                dire_heroes.append(player.hero_id)
        return radiant_heroes, dire_heroes

# A class that processes player data in a match.
class Player:
    def __init__(self, data, api='steampowered'):
        self.account_id = data['account_id']
        self.hero_id = data['hero_id']
        self.kills = data['kills']
        self.deaths = data['deaths']
        self.assists = data['assists']
        self.last_hits = data['last_hits']
        self.denies = data['denies']
        self.gpm = data['gold_per_min']
        self.xpm = data['xp_per_min']

        player_slot = data['player_slot']
        self.team = Team.Dire if (player_slot >> 7) & 1 == 1 else Team.Radiant
        self.slot = player_slot & 0b111

# Fetches the data for a single match in JSON format.
def fetch_match(match_id, api='steampowered', key=None):
    if api == 'steampowered':
        url = f"https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V001/?match_id={match_id}&key={key}"
        r = requests.get(url)
        return json.loads(r.text)['result']
    elif api == 'opendota':
        url = f"https://api.opendota.com/api/matches/{match_id}"
        r = requests.get(url)
        return json.loads(r.text)
    return json.loads("{}")

# Gets a hero's id by their name.
def get_hero_by_name(hero):
    return next((x for x in HEROES.keys() if hero.lower() in [y.lower() for y in HEROES[x]]), 0)

# The main entry point to the program. This program mainly handles data collection and fetching.
def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Dota 2 Match Data")
    parser.add_argument("-api", type=str, default='steampowered', help="The API to use to fetch the data (default is 'steampowered'; can also be 'opendota', however OpenDota has a request limit rate of 60 requests per minute).")
    parser.add_argument("-fetch", action="store_true", default=False, help="Fetches raw match data.")
    parser.add_argument("--start_id", type=int, default=5000000000, help="The match ID to begin fetching data from.")
    parser.add_argument("--N", type=int, default=25, help="The number of matches to fetch (set to 0 to run indefinitely).")
    parser.add_argument("--dir", type=str, default='matches', help="The directory to save the data to (default is 'matches').")
    parser.add_argument("-get", metavar="ID", type=int, help="Fetches data for a single match with the given ID.")
    args = parser.parse_args()

    # Get the API key.
    key = None
    if os.path.exists("keys.json"):
        with open("keys.json", "r") as file:
            keys = json.load(file)
            if args.api in keys:
                key = keys[args.api]

    # Bulk fetching of data.
    if args.fetch:
        api_dir = os.path.join(args.dir, args.api)
        if not os.path.exists(api_dir):
            os.makedirs(api_dir)

        rate_limit = False
        match_id = args.start_id
        while (match_id < args.start_id + args.N) or (args.N == 0):
            path = os.path.join(args.dir, args.api, str(match_id) + ".json")
            if os.path.exists(path):
                print(f"Data for match ID {match_id} already exists. Skipping.")
                match_id += 1
            else:
                data = fetch_match(match_id, api=args.api, key=key)
                error = data['error'] if 'error' in data.keys() else None
                if args.api == 'opendota' and error == 'rate limit exceeded':
                    if rate_limit == False:
                        print(f"Rate limit exceeded for match ID {match_id}. Trying again.")
                        rate_limit = True
                else:
                    rate_limit = False
                    with open(path, encoding='utf-8', mode="w") as file:
                        json.dump(data, file)
                    if error is None:
                        print(f"Fetched data for match ID {match_id} (saved to '{path}').")
                    else:
                        print(f"Failed to retrieve data for match ID {match_id}: {error}")
                    match_id += 1
    
    # Fetch data for a single match.
    if args.get:
        data = fetch_match(args.get, api=args.api, key=key)
        if 'error' in data.keys():
            print(f"Failed to retrieve data for match ID {args.get}: {data['error']}")
            exit()

        match = Match(data)
        print(f"Match ID: {match.match_id} | Duration: {'%.2f mins' % match.duration} | Winner: {'Unknown' if match.winner is None else match.winner}")
        format_mask = "{0:12} | {1:20} | {2:3}/{3:3}/{4:3} | {5:5} | {6:5} | {7:5} | {8:5}"
        print("--- RADIANT ---")
        print(format_mask.format("Player ID", "Hero", "  K", "  D", "  A", "  GPM", "  XPM", "   LH", "   DN"))
        for player in [x for x in match.players if x.team == Team.Radiant]:
            hero = HEROES[player.hero_id][0]
            print(format_mask.format(player.account_id, hero, player.kills, player.deaths, player.assists, player.gpm, player.xpm, player.last_hits, player.denies))
        print("--- DIRE ---")
        for player in [x for x in match.players if x.team == Team.Dire]:
            hero = HEROES[player.hero_id][0]
            print(format_mask.format(player.account_id, hero, player.kills, player.deaths, player.assists, player.gpm, player.xpm, player.last_hits, player.denies))

if __name__ == '__main__':
    main()
