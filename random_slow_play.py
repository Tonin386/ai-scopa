from AI.AIAgent import AIAgent
from Scopa.Game import Game
from AI.AI import AI
import torch

def main():
    state_size = 20
    action_size = 10

    network = AI(state_size, action_size)
    network.load_state_dict(torch.load("models/last_model.pth"))

    agent = AIAgent(state_size, action_size, network)

    game = Game(agent, mode="random", verbose=True)

    better_network = AI(state_size, action_size)
    better_network.load_state_dict(torch.load("models/last_better_model.pth"))

    game.players[0].ai_agent.network = better_network

    for _ in range(10000):
        input("Press key to start next game")
        game.newGame()
        points = game.play()
        print(f"Team1 points: {game.teams[0].points}\nTeam2 points: {game.teams[1].points}")
    

if __name__ == "__main__":
    main()