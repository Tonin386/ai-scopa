import itertools
from Scopa.Player import Player
from Scopa.Card import Card
from Scopa.Team import Team
import random
import cupy as np

FAMILIES = ['GOLDS', 'CUPS', 'SWORDS', 'STICKS']

def do_nothing(args, **kwargs):
    pass

def find_combinations(array, target_sum):
    result = []
    for r in range(1, len(array) + 1):
        for combination in itertools.combinations(array, r):
            if sum([x.value for x in combination]) == target_sum:
                result.append(combination)
    return result

class Game:
    def __init__(self, ai_agent=None, mode="train", main_player=0, train_level="high", verbose=False) -> None:
        self.teams = [Team(0), Team(1)]
        self.players = [Player(self.teams[0], 0, ai_agent=ai_agent), Player(self.teams[1], 1, ai_agent=ai_agent), Player(self.teams[0], 2, ai_agent=ai_agent), Player(self.teams[1], 3, ai_agent=ai_agent)]
        self.deck = [Card(i+1, f) for f in FAMILIES for i in range(10)]
        self.turn_number = 0
        self.start_number = 0
        self.pot = []
        self.mode = mode
        self.main_player = main_player
        self.train_level = train_level
        self.verbose = verbose

    def potToArray(self) -> object:
        num_pot = np.zeros(40)

        for card in self.pot:
            num_pot[int(card)] = 1

        return num_pot
    
    def teamCardsToArray(self) -> tuple:
        team1_cards_num  = np.zeros(40)
        team2_cards_num  = np.zeros(40) #this is a mask I guess

        for card in self.teams[0].won_cards:
            team1_cards_num[int(card)] = 1

        for card in self.teams[1].won_cards:
            team2_cards_num[int(card)] = 1

        return team1_cards_num, team2_cards_num

    def resetDeck(self) -> None:
        self.deck = [Card(i+1, f) for f in FAMILIES for i in range(10)]

    def shuffleDeck(self) -> None:
        random.shuffle(self.deck)

    def dealCards(self) -> None:
        if self.mode != "cheat":
            for i, player in enumerate(self.players):
                player.getCards(self.deck[i::4])
        else:
            self.resetDeck()
            cards = []
            for i in range(10):
                print("Families: ", FAMILIES)
                print("Card values: ", [i for i in range(1, 10)])
                family = FAMILIES[int(input("Pick a family for your card: "))]
                value  = int(input("Pick a value for your card: "))
                card = self.deck.pop(self.deck.index(Card(value, family)))
                cards.append(card)

            self.players[self.main_player].getCards(cards)

    def isOver(self) -> bool:
        return not any([x.hasCards() for x in self.players])

    def managePot(self, chosen_combination=0) -> list:
        if len(self.pot) < 2:
            return []

        last_card = self.pot[-1]
        won_cards = []
        if last_card.value in [x.value for x in self.pot[:-1]]:
            idx = [x.value for x in self.pot[:-1]].index(last_card.value)
            won_cards.extend([self.pot.pop(idx), self.pot.pop()])
        else:
            combinations = find_combinations(self.pot[:-1], last_card.value)
            if len(combinations) == 0:
                return []
            
            for card in combinations[chosen_combination]:
                won_cards.append(self.pot.pop(self.pot.index(card)))
            won_cards.append(self.pot.pop())

        return won_cards

    def isScopa(self) -> bool:
        return len(self.pot) == 0
    
    def countPoints(self) -> None:
        if not self.verbose:
            print = do_nothing

        team1_cards = self.teams[0].won_cards
        team2_cards = self.teams[1].won_cards
        team1_wonPoints = 0
        team2_wonPoints = 0
        print("Team1 cards: %s" % str([str(x) for x in team1_cards]))
        print("Team2 cards: %s" % str([str(x) for x in team2_cards]))

        #Who has more cards?
        if len(team1_cards) != len(team2_cards):
            if len(team1_cards) > len(team2_cards):
                team1_wonPoints += 1
                print("Team1 won cards")
            else:
                team2_wonPoints += 1
                print("Team2 won cards")

        #Who has more golds?
        team1_golds = sum([1 if card.family == "GOLDS" else 0 for card in team1_cards])
        team2_golds = sum([1 if card.family == "GOLDS" else 0 for card in team2_cards])
        if team1_golds != team2_golds:
            if team1_golds > team2_golds:
                team1_wonPoints += 1
                print("Team1 won golds")
            else:
                team2_wonPoints += 1
                print("Team2 won golds")

        #Who made primera? cazzo
        team1_primera_cards = {family: (0,) for family in FAMILIES}
        team1_primera_value = 0
        team2_primera_cards = {family: (0,) for family in FAMILIES}
        team2_primera_value = 0

        #check for team1
        for card in team1_cards:
            if card.value in [1, 6, 7]:
                if max(team1_primera_cards[card.family]) < card.value:
                    team1_primera_cards[card.family] = (card.value,)

        if all([team1_primera_cards[family][0] > 0 for family in team1_primera_cards]): #has at least 1 card in every family
            team1_primera_value = sum([team1_primera_cards[family][0] for family in team1_primera_cards])

        #check for team2
        for card in team2_cards:
            if card.value in [1, 6, 7]:
                if max(team2_primera_cards[card.family]) < card.value:
                    team2_primera_cards[card.family] = (card.value,)

        if all([team2_primera_cards[family][0] > 0 for family in team2_primera_cards]): #has at least 1 card in every family
            team2_primera_value = sum([team2_primera_cards[family][0] for family in team2_primera_cards])

        if team1_primera_value != team2_primera_value:
            if team1_primera_value > team2_primera_value:
                team1_wonPoints += 1
                print("Team1 won primera")
            else:
                team2_wonPoints += 1
                print("Team2 won primera")

        #Did someone make napola? How much points?
        found = True
        points = 0
        while found:    
            found = Card(points+1, "GOLDS") in team1_cards
            if found:
                points += 1

        if points >= 3:
            team1_wonPoints += points
            print("Team1 made napola (%d)" % points)

        else:
            found = True
            points = 0
            while found:    
                found = Card(points+1, "GOLDS") in team2_cards
                if found:
                    points += 1
                
            if points >= 3:
                team2_wonPoints += points
                print("Team2 made napola (%d)" % points)

        #Who has Settebello?
        if Card(7, "GOLDS") in team1_cards:
            team1_wonPoints += 1
            print("Team1 won Settebello")
        else:
            team2_wonPoints += 1
            print("Team2 won Settebello")

        self.teams[0].points += team1_wonPoints
        self.teams[1].points += team2_wonPoints

        return team1_wonPoints, team2_wonPoints
    
    def play(self) -> None:
        last_taker = 0
        while not self.isOver():
            reward = 0
            current_state = self.getState()
            current_state = np.reshape(current_state, [1, 20])
            if self.mode != "cheat":
                if self.mode == "true-random":
                    played_card = self.players[self.turn_number].playCard()
                elif self.train_level == "low":
                    played_card = self.players[self.turn_number].playCard() if self.players[self.turn_number].id != self.main_player else self.players[self.turn_number].playCard(human_pick=False, random_pick=False, state=current_state)
                else:
                    if self.turn_number == self.main_player:
                        # print("Better AI hand is: ", [str(card) for card in self.players[0].cards])
                        pass
                    played_card = self.players[self.turn_number].playCard(human_pick=False, random_pick=False, state=current_state)
            else:
                if self.turn_number != self.main_player:
                    print(f"You have to say what player #{self.turn_number} has played.")
                    played_card = self.players[self.turn_number].playCard(human_pick=True, random_pick=False)
                else:
                    played_card = self.players[self.turn_number].playCard(human_pick=False, random_pick=False, state=current_state)

            self.pot.append(played_card[0])
            won_cards = self.managePot()
            if len(won_cards) >= 1:
                last_taker = self.turn_number
            self.players[self.turn_number].team.won_cards.extend(won_cards)
            if self.verbose:
                print("Player %s plays %s. Pot is now %s" % (self.players[self.turn_number], played_card[0], [str(card) for card in self.pot]))

            if not self.isOver() and self.isScopa():
                reward += 1
                self.players[self.turn_number].team.points += 1
            
            if not self.isOver() and self.turn_number == self.main_player:
                reward += len(won_cards) / 40
                new_state = self.getState()
                new_state = np.reshape(new_state, [1, 20])
                if self.mode == "train":
                    self.players[self.main_player].ai_agent.remember(current_state, played_card[1], reward, new_state, self.isOver())

            self.turn_number = (self.turn_number + 1) % 4

        self.players[last_taker].team.won_cards.extend(self.pot)
        reward_points = self.countPoints()
        game_reward = reward_points[self.players[self.main_player].team.id]
        # print("Reward for this game:", game_reward)
        new_state = self.getState()
        new_state = np.reshape(new_state, [1, 20])
        if self.mode == "train":
            self.players[self.main_player].ai_agent.remember(current_state, played_card[1], reward, new_state, True)
            self.players[self.main_player].ai_agent.replay(game_reward)

        return reward_points

    def getState(self) -> object:
        pot_array =  [int(c) for c in self.pot] + list([-1] * (10 - len(self.pot)))
        player_hand = list(self.players[0].getHandArray()) + list([-1] * (10 - len(self.players[0].getHandArray())))
        state = np.array(pot_array + player_hand)
        #Team cards?
        return state

    def newGame(self) -> None:
        self.pot = []
        self.teams[0].won_cards = []
        self.teams[1].won_cards = []
        self.start_number = (self.start_number + 1) % 4 
        self.turn_number = self.start_number
        self.resetDeck()
        self.shuffleDeck()
        self.dealCards()
        return self.getState()