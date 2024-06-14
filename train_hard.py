import time
import torch
from tqdm import tqdm
from Scopa.Game import Game
from AI.AIAgent import AIAgent
from AI.AI import AI

def main():
    state_size = 20
    action_size = 10


    network = AI(state_size, action_size)
    network.load_state_dict(torch.load("models/last_model.pth"))

    agent = AIAgent(state_size, action_size,network=network)

    game = Game(agent, mode="train", train_level="high")

    episodes = 10000
    start = time.time()
    for e in tqdm(range(episodes)):
        game.newGame()
        game.play()

    end = time.time()
    print("Simulated %d games in %.4fs" % (episodes, end-start))
    print(f"Team1 points: {game.teams[0].points}\nTeam2 points: {game.teams[1].points}")
    
    torch.save(agent.network.state_dict(), 'models/last_better_model.pth')

if __name__ == "__main__":
    main()