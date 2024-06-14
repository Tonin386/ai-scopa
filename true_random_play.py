from AI.AIAgent import AIAgent
from Scopa.Game import Game
from AI.AI import AI
import torch
from tqdm import tqdm

def main():
    state_size = 20
    action_size = 10

    agent = AIAgent(state_size, action_size)

    game = Game(agent, mode="true_random")

    total_wins = 0
    games = 1000

    for _ in tqdm(range(games)):
        game.newGame()
        points = game.play()
        total_wins += 1 if points[0] > points[1] else 0
    
    print(f"Team1 points: {game.teams[0].points}\nTeam2 points: {game.teams[1].points}")
    print(f"Team1 won {total_wins * 100 / games}% of the games.")

if __name__ == "__main__":  
    main()