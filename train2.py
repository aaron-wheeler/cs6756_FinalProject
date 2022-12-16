import torch
import numpy as np
import math
import gym
import argparse
import random

# plotting
import matplotlib.pyplot as plt

# ML_pipeline utilities
from ML_pipeline.my_classes import *

# simulator utilities
from gym_dpr.envs.viz import Visualizer
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World
import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot

# additional data methods for IL
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import ConcatDataset

# for DQN
from random import random, randint, sample
from collections import deque

def main(args):
    if args.generate:
        #----- Expert Data Generation Step -----#
        print('=' * 40)
        print("Initiating Behavioral Cloning Rollouts")
        print('=' * 40)

        # initialize
        num_trajectories = 100
        expert_states = []
        expert_actions = []
        expert_features = []
        D_collection = []

        # Create environment
        env = gym.make('dpr_single-v0',
                    numBots=9, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
                    discreteActionSpace=False, continuousAction=False,
                    goalFrame=True,
                    rewardFunc="piecewise",
                    randomSeed=0,
                    fixedStart=False, fixedGoal=True,
                    fixedStartCoords=None, fixedGoalCoords=(0, 0),
                    polarStartCoords=False, polarGoalCoords=False,
                    transformRectStart=(0, 0), transformRectGoal=(0, 0),
                    xLower=-300, xUpper=300, yLower=-300, yUpper=300,
                    radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
                    numDead=0, deadIxs=None,
                    gate=False, gateSize=150,
                    manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                    # visualizer=Visualizer(), recordInfo=True)
                    visualizer=None, recordInfo=False)

        # generate rollouts
        for rollout in range(num_trajectories):
            print(f"Trajectory: [{rollout+1}/{num_trajectories}]")
            states_list = []
            actions_list = []

            obs = env.reset()
            print("starting position = ", env.start)
            while True:
                # hand crafted wave policy
                totalSteps, actions = env.wavePolicy() # a single expansion cycle
                for i in range(totalSteps):
                    # step env
                    for _ in range(10):
                        action = actions[i]

                        # store obs-action pairs
                        states_list.append(obs)
                        actions_list.append(action)

                        env.render()
                        obs, reward, done, info = env.step(action)
                if done:
                    print("final position = ", env.agent.getCOM())
                    print('-' * 20)
                    break
            env.close()
            # transform and append trajectories
            states_list = np.array(states_list)
            actions_list = np.array(actions_list)
            expert_states.append(states_list)
            expert_actions.append(actions_list)

        # convert numpy array to torch tensor
        expert_states = np.concatenate(expert_states)
        expert_actions = np.concatenate(expert_actions)
        inputs = torch.from_numpy(expert_states).type(torch.float)
        outputs = torch.from_numpy(expert_actions).type(torch.float)
        # create a Tensor Dataset and append it to D_collection
        ExpertTensorDataset = TensorDataset(inputs, outputs)
        D_collection.append(ExpertTensorDataset)
        init_D_collection = ConcatDataset(D_collection)
        n_pairs = len(init_D_collection)
        print("n_pairs = ", n_pairs)

        # prepare dataloader
        dataloader = DataLoader(
            init_D_collection,
            shuffle=False,
            batch_size=64
        )

        # save the dataset
        torch.save(dataloader, "data/expert_rollouts_full_100traj.pt")
        print("Saved Dataloader State to data/expert_rollouts_full_100traj.pt")
        # torch.save(dataloader, "data/expert_rollouts_36feats_9out_100traj.pt")
        # print("Saved Dataloader State to data/expert_rollouts_36feats_9out_100traj.pt")
        # torch.save(dataloader, "data/expert_rollouts_36feats_50traj.pt")
        # print("Saved Dataloader State to data/expert_rollouts_36feats_50traj.pt")

    # ====================================================================== #
    
    if args.train_BC:
        #----- Train Behavior Cloning Policy -----#
        print('=' * 40)
        print("Initiating Training of Behavioral Cloning Policy")
        print('=' * 40)

        # load pre-generated dataloader
        # dataloader = torch.load("data/expert_rollouts_36feats_9out_100traj.pt")
        dataloader = torch.load("data/expert_rollouts_full_100traj.pt")
        batch_size = 64

        # prepare plotting utilities
        iters, BC_train_loss = [], []
        # steps, train_loss = [], []

        # instantiate net and create initial policy
        model = ImitationNet(features=36)
        optim = torch.optim.Adam(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.5)
        ratio = 3 # 25218 samples -> 18913.5 ==0 / 6304.5==1 per class (9 classes)
        pos_weight = torch.full((9,), 3)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_steps = 5000
        n_check = 500
        n = 0 # the number of iterations
        for step in range(n_steps):
            # batch-wise training for BC method
            for batch, (X, y) in enumerate(dataloader):
                if len(y) != batch_size:
                    continue
                episode_loss = learn(model, loss_fn, optim, X, y)

                # save training info for plotting
                iters.append(n)
                BC_train_loss.append(episode_loss / batch_size)
                n += 1

            scheduler.step()
            if step % n_check == 0:
                print(f"BC Epoch:  [{(step+n_check):>5d}/{n_steps:>5d}], training_loss: {episode_loss:>7f}")
                torch.save(model.state_dict(), "BC_model/model_BC_{}_5000.pth".format(step+n_check))

        print("BC step complete")

        # plot training loss vs. iterations
        plt.title(f"BC Loss Curves")
        plt.plot(iters, BC_train_loss, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig('figs2/BC_loss.png')
        plt.close()

        # save the trained model
        torch.save(model.state_dict(), "BC_model/model_BC.pth")
        print("Saved PyTorch Model State to BC_model/model_BC.pth")

    # ====================================================================== #
    
    if args.train_DQN:
        #----- Train DQN Policy -----#

        # Create environment
        env = gym.make('dpr_single-v0',
                    numBots=9, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
                    discreteActionSpace=False, continuousAction=False,
                    goalFrame=True,
                    rewardFunc="piecewise",
                    randomSeed=0,
                    fixedStart=False, fixedGoal=True,
                    fixedStartCoords=None, fixedGoalCoords=(0, 0),
                    polarStartCoords=False, polarGoalCoords=False,
                    transformRectStart=(0, 0), transformRectGoal=(0, 0),
                    xLower=-300, xUpper=300, yLower=-300, yUpper=300,
                    radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
                    numDead=0, deadIxs=None,
                    gate=False, gateSize=150,
                    manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                    # visualizer=Visualizer(), recordInfo=True)
                    visualizer=None, recordInfo=False)

        total_reward = 0
        done = False
        state = env.reset()
        
        # neural net utilities
        model_DQN = DQN(36)

        # Transfer learning/Warmstarting Model
        # model_DQN.load_state_dict(torch.load("models_BC/model_BC_weight_nav_3000_10000.pth"))
        model_DQN.load_state_dict(torch.load("BC_model/model_BC.pth"))

        optim = torch.optim.Adam(model_DQN.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)
        ratio = 3 # 25218 samples -> 18913.5 ==0 / 6304.5==1 per class (9 classes)
        pos_weight = torch.full((9,), 3)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        batch_size = 64 #16 # 256 
        gamma = 0.99

        # Number of epoches between testing phases
        replay_memory_size = 24000 # 240000
        replay_memory = deque(maxlen=replay_memory_size)

        # Exploration or exploitation parameters
        initial_epsilon = 1
        final_epsilon = 0.065 # 1e-3
        num_decay_epochs= 10 # exploration_fraction=0.1 -> "fraction of entire training period over which the exploration rate is reduced"

        # prepare plotting utilities
        steps, train_loss, rewards = [], [], []
        step = 0
        k = 0
        j = 0
        i = 0

        # Initiate training loop
        epoch = 0
        num_epochs = 400 # 1000000
        e_update = 40 # 40000
        while epoch < num_epochs:
            # Exploration or exploitation
            epsilon = final_epsilon + (max(num_decay_epochs - epoch, 0) * (
                    initial_epsilon - final_epsilon) / num_decay_epochs)
            u = random()
            random_action = u <= epsilon

            # adjust ob type to work w Neural Network
            obs = torch.from_numpy(state).type(torch.float)

            model_DQN.eval()
            with torch.no_grad():
                prediction = model_DQN(obs)
            model_DQN.train()
            if random_action:
                action = env.action_space.sample()
            else:
                action = torch.round(torch.sigmoid(prediction)) # action that maximizes Q*
                action = action.detach().numpy()

            env.render()
            next_state, reward, done, info = env.step(action)
            next_obs = torch.from_numpy(next_state).type(torch.float)
            total_reward += reward

            replay_memory.append([obs, reward, next_obs, done])
            if done:
                final_score = total_reward
                total_reward = 0
                state = env.reset()
                k += 1
            else:
                state = next_state
                j += 1
                continue
            if len(replay_memory) < replay_memory_size / 10:
                i += 1
                continue
            epoch += 1
            batch = sample(replay_memory, min(len(replay_memory), batch_size))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = torch.stack(tuple(state for state in state_batch))
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.stack(tuple(state for state in next_state_batch))

            # Optimization step
            q_values = model_DQN(state_batch) # Q(s_{t}) for each action, according to current policy
            model_DQN.eval()
            with torch.no_grad():
                next_prediction_batch = model_DQN(next_state_batch) # Q(s_{t+1}) for each action, according to current policy
            model_DQN.train()

            target_list = []
            for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch):
                if done == True:
                    target_list.append(torch.full((9,), reward.item()))
                else:
                    target_list.append(torch.add(torch.mul((torch.round(torch.sigmoid(prediction))), gamma), reward))
            
            Q_target = torch.stack(target_list)

            Q_sa = torch.round(torch.sigmoid(q_values))

            optim.zero_grad()
            loss = criterion(Q_sa, Q_target)
            loss.backward()
            optim.step()
            scheduler.step()

            # save training info for plotting
            step += 1
            steps.append(step)
            train_loss.append(loss.item())
            rewards.append(final_score)

            if epoch % e_update == 0:
                print("Epoch: [{}/{}], Loss: {}, Score: {}".format(
                    (epoch),
                    num_epochs,
                    loss.item(),
                    final_score))
                torch.save(model_DQN.state_dict(), "models_RL/model_DQN_warm2_{}_400.pth".format(epoch+e_update))

        print("DQN training complete!")

        # debug
        print("number of simulation [*done] = ", k)
        print("number of state-action pairs [*next state] = ", j) # 250000
        print("number below buffer [*passed replay] = ", i)

        # plot batch training loss vs. number of optimization steps
        # plt.title("DQN Loss Curves")
        plt.title("DQN Loss Curves (Pre-trained)") # TRANSFER version
        plt.plot(steps, train_loss, label="Batch Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        # plt.savefig('figs/DQN_loss.png')
        plt.savefig('figs2/DQN_warm2_loss.png') # TRANSFER version
        plt.close()

        # plot reward vs. number of optimization steps
        # plt.title("DQN Total Rewards")
        plt.title("DQN Total Rewards (Pre-trained)") # TRANSFER version
        plt.plot(steps, rewards, label="Reward")
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.legend(loc='best')
        # plt.savefig('figs/DQN_reward.png')
        plt.savefig('figs2/DQN_warm2_reward.png') # TRANSFER version
        plt.close()

        # save the trained model
        env.close()
        # torch.save(model_DQN.state_dict(), "models_RL/model_DQN.pth")
        # print("Saved PyTorch Model State to models_RL/model_DQN.pth")
        torch.save(model_DQN.state_dict(), "models_RL/model_DQN_warm2.pth") # TRANSFER version
        print("Saved PyTorch Model State to models_RL/model_DQN_warm2.pth") # TRANSFER version


    # ====================================================================== #
    
    elif args.eval:
        #----- Evaluate Trained Policy -----#
        print('=' * 40)
        print("Initiating Evaluation of Trained Policy")
        print('=' * 40)

        # REGISTER ENV
        env = gym.make('dpr_single-v0',
                    numBots=9, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
                    discreteActionSpace=False, continuousAction=False,
                    goalFrame=True,
                    rewardFunc="piecewise",
                    randomSeed=0,
                    fixedStart=False, fixedGoal=True,
                    fixedStartCoords=None, fixedGoalCoords=(0, 0),
                    # fixedStart=True, fixedGoal=True,
                    # fixedStartCoords=(254,254), fixedGoalCoords=(0, 0),
                    polarStartCoords=False, polarGoalCoords=False,
                    transformRectStart=(0, 0), transformRectGoal=(0, 0),
                    xLower=-300, xUpper=300, yLower=-300, yUpper=300,
                    radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
                    numDead=0, deadIxs=None,
                    gate=False, gateSize=150,
                    manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                    # visualizer=Visualizer(), recordInfo=True)
                    visualizer=None, recordInfo=False)
        # check_env(env)

        # load the trained model
        print("Loading model...")
        # model = ImitationNet(features=36)
        # model.load_state_dict(torch.load("models/model_BC_weight_nav.pth"))
        # model.load_state_dict(torch.load("models_BC/model_BC_weight_nav_3000_10000.pth"))

        # model = DQN(36)
        # model.load_state_dict(torch.load("models_RL/model_DQN.pth"))
        # model.load_state_dict(torch.load("models_RL/model_DQN_warm.pth"))
        # model.load_state_dict(torch.load("models_RL/model_DQN_warm2.pth"))

        # model.load_state_dict(torch.load("BC_model/model_BC.pth"))
        # model.load_state_dict(torch.load("models_RL/model_DQN_warm2.pth"))
        print("Model loaded. Initiating rollout now...")

        # Log total score in each episode
        num_episodes = 10
        ep_scores = np.zeros(num_episodes)
        ep_dists = np.zeros(num_episodes)

        for episode in range(0, num_episodes):
            # TEST THE POLICY AND RENDER
            obs = env.reset()
            score = 0
            dist = 0
            print("starting position = ", env.start)
            xs, ys = env.start[0], env.start[1]
            while True:
                totalSteps, actions = env.wavePolicy() # a single expansion cycle
                for i in range(totalSteps):
                    # step env
                    for _ in range(10):
                        action = actions[i]
                        env.render()
                        obs, reward, done, info = env.step(action)
                        score += reward

                # # adjust ob type to work w Neural Network
                # obs = torch.from_numpy(obs).type(torch.float)
                # # make prediction
                # model.eval()
                # with torch.no_grad():
                #     prediction = model(obs)
                # action = torch.round(torch.sigmoid(prediction))
                # action = action.detach().numpy()

                # obs, reward, done, info = env.step(action)
                # env.render()
                # score += reward

                if done:
                    print("final position = ", env.agent.getCOM())
                    fin_pos = env.agent.getCOM()
                    xf, yf = fin_pos[0], fin_pos[1]
                    dist = math.sqrt((xf - xs)**2 + (yf - ys)**2)
                    print("distance traveled", dist)
                    # print('-' * 20)
                    break
            env.close()

            ep_scores[episode] = score
            ep_dists[episode] = dist
            print(f"Episode: {episode}  Total score: {ep_scores[episode]}")
            print(f"Episode: {episode}  Total distance: {ep_dists[episode]}")
            print('-' * 20)

        print(f"Summary over {num_episodes} epsiodes:  Highest score: {np.max(ep_scores)} Mean score: {np.mean(ep_scores)} Std score: {np.std(ep_scores)}")
        print(f"Summary over {num_episodes} epsiodes:  Highest distance: {np.max(ep_dists)} Mean distance: {np.mean(ep_dists)} Std distance: {np.std(ep_dists)}")
    # ====================================================================== #
    # #----- Test Methods -----#
    if args.test_data:
        # load pre-generated dataloader
        # dataloader = torch.load("data/expert_rollouts_36feats_9out_100traj.pt")
        dataloader = torch.load("data/expert_rollouts_full_100traj.pt")
        print("number of batches = ", len(dataloader))
        print("number of sample pairs = ", len(dataloader.dataset))

        # test data sampling methods
        iterloader = iter(dataloader)
        batch = next(iterloader)
        print("batch = ", batch)
        X, y = batch
        print("X = ", X) # observations
        print("y = ", y) # action targets
        print("y len = ", len(y)) # action targets


        # 1 trajectories -> 253 sample pairs
        # 10 trajectories -> 2520 sample pairs
        # 50 trajectories -> 12605 sample pairs
        # (100 trajectories -> 25218 sample pairs)

        pass

def get_args():
    parser = argparse.ArgumentParser(description='main script')
    parser.add_argument('--generate', action='store_true', help='whether to generate expert trajectories or not')
    parser.add_argument('--train_BC', action='store_true', help='whether to train BC policy or not')
    parser.add_argument('--train_DQN', action='store_true', help='whether to train DQN policy or not')
    parser.add_argument('--eval', action='store_true', help='whether to evaluate trained policy or not')
    parser.add_argument('--test_data', action='store_true', help='whether to evaluate trained policy or not')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(get_args())