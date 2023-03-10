import numpy as np
import pymunk
import pygame
import math
from statistics import mean

import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot


class SuperCircularBot:
    '''A collection of particle robots that can be observed and controleld in a centralized manner'''
    def __init__(self, numBots, pos, botType=CircularBot,
                 continuousAction=False, deadIxs=[]):
        dims = math.ceil(math.sqrt(numBots))

        def posFunc(centerPos, ix):
            BOT_DIAMETER = 2 * DPR_ParticleRobot.BOT_RADIUS + DPR_ParticleRobot.PADDLE_WIDTH + DPR_ParticleRobot.PADDLE_LENGTH
            xc, yc = centerPos
            x = ((ix % dims) - (dims / 2) + 0.5) * BOT_DIAMETER + xc
            y = ((ix // dims) - (dims / 2) + 0.5) * BOT_DIAMETER + yc
            return (x, y)

        particleRobots = []
        for i in range(numBots):
            isDead = (i in deadIxs)
            bot = botType(posFunc(pos, i), i, continuousAction=continuousAction, dead=isDead)
            particleRobots.append(bot)

        self.numBots = numBots
        self.particles = particleRobots

        self.prevCOM = pos
        self.currCOM = pos

    def observeSelf(self, goal):
        '''
        Superagent observes positions and velocities of individual particles
        bot position transformed to goal reference frame

        :return: np array containing x y components of position and velocity and angle for each particle
        '''
        # # EDITED -> Added update call (LINE 46)
        # self.updateCOM()

        xs = []
        ys = []
        vxs = []
        vys = []

        for bot in self.particles:
            x, y, vx, vy = bot.observeSelf()
            # EDITED -> commented out goal adjusted lines (LINE 61)
            x -= goal[0]
            y -= goal[1]
            xs.append(x)
            ys.append(y)
            vxs.append(vx)
            vys.append(vy)

        return np.array(xs + ys + vxs + vys)


    def actionAll(self, action):
        '''
        Superagent controls each particle robot - single agent system with multi-binary action space

        :param action: self.numBots sized array corresponding to action for each particle (binary)
        :return: list with the returns of each action from CircularBot action
        '''
        results = []
        for i in range(self.numBots):
            result = self.particles[i].act(action[i])
            results.append(result)
        return results

    def getCOM(self):
        '''
        Calculates center of mass of the system of particle robots - assumes every particle robot weighs the same

        :return: vector coordinates of center of mass
        '''
        xs = []
        ys = []
        for bot in self.particles:
            x, y = bot.shape.body.position
            xs.append(x)
            ys.append(y)
        return pymunk.vec2d.Vec2d(mean(xs), mean(ys))

    def updateCOM(self):
        '''
        Updates superagent center of mass

        :return:
        '''
        self.prevCOM = self.currCOM
        self.currCOM = self.getCOM()
