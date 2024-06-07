from Scopa.Card import Card
import random

FAMILIES = ['GOLDS', 'CUPS', 'SWORDS', 'STICKS']

class Player:
    def __init__(self, team=None, id=-1, ai_agent=None) -> None:
        self.id = id
        self.cards = []
        self.team = team
        self.ai_agent = ai_agent

    def __str__(self) -> str:
        return "%d" % self.id

    def getCards(self, cards: list) -> None:
        self.cards = cards

    def playCard(self, idx=0, random_pick=True, human_pick=False, **kwargs) -> Card:
        if random_pick and not human_pick:
            idx = random.randint(0, len(self.cards) - 1)
        elif not random_pick and human_pick:
            print("Families: ", FAMILIES)
            print("Card values: ", [i for i in range(1, 11)])
            family = FAMILIES[int(input("Pick a family for your card: "))]
            value  = int(input("Pick a value for your card: "))
            card = Card(value, family)
            idx = -1
        else:
            action = self.ai_agent.act(state=kwargs["state"], hand=self.getHandArray())
            idx = self.cards.index(int(action))
            # print("AI played a card.")

        card = self.cards.pop(idx)
        return card, idx
    
    def hasCards(self) -> bool:
        return len(self.cards) > 0
    
    def __eq__(self, value: float) -> bool:
        return value == self.id
    
    def getHandArray(self) -> object:
        return [int(card) for card in self.cards]