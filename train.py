import time
import torch
from tqdm import tqdm
from Scopa.Game import Game
from AI.AIAgent import AIAgent

def main():
    state_size = 20
    action_size = 10
    agent = AIAgent(state_size, action_size)

    game = Game(agent, mode="train")

    episodes = 10000
    start = time.time()
    for e in tqdm(range(episodes)):
        # input("Press key to start next game")
        game.newGame()
        game.play()
        # print(f"Team1 points: {game.teams[0].points}\nTeam2 points: {game.teams[1].points}")

    end = time.time()
    print("Simulated 1000 games in %.4fs" % (end-start))
    print(f"Team1 points: {game.teams[0].points}\nTeam2 points: {game.teams[1].points}")
    
    torch.save(agent.network.state_dict(), 'models/last_model.pth')

if __name__ == "__main__":
    main()