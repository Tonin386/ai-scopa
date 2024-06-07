FAMILIES = ['GOLDS', 'CUPS', 'SWORDS', 'STICKS']

class Card:
    def __init__(self, value=1, family="GOLDS") -> None:
        self.value = value
        self.family = family

    def __eq__(self, card: object) -> bool:
        if isinstance(card, Card):
            return card.value == self.value and card.family == self.family
        return int(self) == card
    
    def __str__(self) -> str:
        return str(self.value) + " of " + self.family
    
    def __int__(self) -> int:
        # 0 means no card in array
        return 10 * FAMILIES.index(self.family) + self.value % 10
