import random
import pygame
from itertools import cycle
import os


class GameState:

    def __init__(self, headless=False):

        self.headless = headless
        if headless == True:
            os.putenv('SDL_VIDEODRIVER', 'dummy')

        self.FPS = 99999999999999999999#30
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREENWIDTH  = 288
        self.SCREENHEIGHT = 512

        pygame.init()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        self.OFFSCREEN = pygame.Surface((self.SCREENWIDTH, self.SCREENHEIGHT), flags=4, depth=32)

        #SCREEN = pygame.display.set_mode((1, 1))
        #pygame.display.set_caption('Flappy Bird')

        self.load_assets()
        self.reset()


    def reset(self):
        self.PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
        self.BASEY = self.SCREENHEIGHT * 0.79

        self.PLAYER_WIDTH = self.IMAGES['player'][0].get_width()
        self.PLAYER_HEIGHT = self.IMAGES['player'][0].get_height()
        self.PIPE_WIDTH = self.IMAGES['pipe'][0].get_width()
        self.PIPE_HEIGHT = self.IMAGES['pipe'][0].get_height()
        self.BACKGROUND_WIDTH = self.IMAGES['background'].get_width()

        self.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(self.SCREENWIDTH * 0.2)
        self.playery = int((self.SCREENHEIGHT - self.PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.BACKGROUND_WIDTH

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps


    def load_assets(self):
        # path of player with different states
        PLAYER_PATH = (
                'assets/sprites/redbird-upflap.png',
                'assets/sprites/redbird-midflap.png',
                'assets/sprites/redbird-downflap.png'
        )

        # path of background
        BACKGROUND_PATH = 'assets/sprites/background-black.png'

        # path of pipe
        PIPE_PATH = 'assets/sprites/pipe-green.png'

        self.IMAGES, self.HITMASKS = {}, {}

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        # select random background sprites
        self.IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

        # select random player sprites
        self.IMAGES['player'] = (
            pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
            pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
            pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
        )

        # select random pipe sprites
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPE_PATH).convert_alpha(), 180),
            pygame.image.load(PIPE_PATH).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )


    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask


    def frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # check for score
        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash = self.checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            terminal = True
            self.reset()
            reward = -1

        # draw sprites
        self.OFFSCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.OFFSCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.OFFSCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.OFFSCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))

        # print score so player overlaps the score
        # showScore(self.score)
        self.OFFSCREEN.blit(self.IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(self.OFFSCREEN)

        if self.headless == False:
            self.SCREEN.blit(self.OFFSCREEN, (0, 0))
            pygame.display.update()
            self.FPSCLOCK.tick(self.FPS)

        return image_data, reward, terminal


    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs)-1)
        gapY = gapYs[index]

        gapY += int(self.BASEY * 0.2)
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - self.PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]


    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.OFFSCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()


    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                          player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False


    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False
