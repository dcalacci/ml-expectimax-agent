from util import manhattanDistance
from game import Directions
import math
import random, util, mazeUtils
from game import Agent


_a = _b = _c = _d = _e = _f = 0.0
def storeConstants(a, b, c, d, e, f):
  global _a, _b, _c, _d, _e, _f
  _a = float(a)
  _b = float(b)
  _c = float(c)
  _d = float(d)
  _e = float(e)
  _f = float(f)

_walls = None
distanceInMaze = {}
def computeMazeDistances(walls):
  global _walls
  if walls == _walls:
    return
  else:
    _walls = walls
    mazeUtils.distancesInMaze(walls, distanceInMaze)

def getDistanceInMaze(start, goal):

  def floor(pos):
    return (math.floor(pos[0]), math.floor(pos[1]))

  start = floor(start)
  goal = floor(goal)
  sg = (start, goal)
  gs = (goal, start)
  if distanceInMaze.has_key(sg):
    return distanceInMaze[sg]
  elif distanceInMaze.has_key(gs):
    return distanceInMaze[gs]
  else:
    raise Exception("no distance stored for that pair")

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  
    
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.
    
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    
    "Add more of your code here if you want to"
    
    return legalMoves[chosenIndex]
  
  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here. 
    
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    
    The code below extracts some useful information from the state, like the 
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    walls = currentGameState.getWalls()
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates() 
    ghostPositions = map(lambda g: g.getPosition(), newGhostStates)
#    computeMazeDistances(walls)

    # getting closer to food is good
    # getting closer to ghosts is bad

    foodScore = 0
#    distanceToClosestFood = min(map(lambda x: getDistanceInMaze(newPos, x), oldFood.asList()))
    distanceToClosestFood = min(map(lambda x: util.manhattanDistance(newPos, x), oldFood.asList()))

    distanceToClosestGhost = min(map(lambda x: util.manhattanDistance(newPos, x), 
                                     ghostPositions))

    ghostScore = 0
    foodScore = 0
    if distanceToClosestGhost == 0:
      return -99
    elif distanceToClosestGhost < 6:
      ghostScore = (1./distanceToClosestGhost) * -2
    
    if distanceToClosestFood == 0:
      foodScore = 0
      ghostScore += 2
    else:
      foodScore = 1./distanceToClosestFood

    return foodScore + ghostScore

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()



class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.  
  """
  
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', a=0,b=0,c=0,d=0,e=0,f=0):
    self.index = 0 # Pacman is always agent index 0
    self.depth = int(depth)
    storeConstants(a, b, c, d, e, f)
    # self.a = a
    # print "a: ", a
    # b = b
    # c = c
    # d = d
    # e = e
    # f = f
    self.evaluationFunction = util.lookup(evalFn, globals())

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def gameOver(self, gameState, d):
    return gameState.isLose() or gameState.isWin() or d == 0

  def minimax(self, gameState, agentIndex, depth):
    "produces the min or max value for some game state and depth; depends on what agent."
    successorStates = map(lambda a: gameState.generateSuccessor(agentIndex, a),
                         gameState.getLegalActions(agentIndex))
    if self.gameOver(gameState, depth): # at an end
      return self.evaluationFunction(gameState)
    else:
      # use modulo so we can wrap around, get vals of leaves
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      vals = map(lambda s: self.minimax(s, nextAgent, depth - 1), 
                successorStates)      
      if nextAgent == 0: # pacman
        return max(vals)
      else:
        return min(vals)
  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth 
      and self.evaluationFunction.
    """
    depth = gameState.getNumAgents()*self.depth

    legalActions = gameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    successorStates = map(lambda a: gameState.generateSuccessor(0,a),
                          legalActions)
    # compute the best possible values for each successorState
    vals = map(lambda s: self.minimax(s, 1, depth - 1), 
               successorStates)
    # return the action that corresponds to the largest value
    return legalActions[vals.index(max(vals))]

    
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def gameOver(self, gameState, d):
    return gameState.isLose() or gameState.isWin() or d == 0

  def alphabeta(self, agentIndex, gameState, depth, alpha, beta):
    "alpha-beta search. pretty similar to minimax, but it prunes the tree etc."
    legalActions = gameState.getLegalActions(agentIndex)
    if agentIndex == 0 and Directions.STOP in legalActions:
      legalActions.remove(Directions.STOP)

    successorStates = map(lambda a: gameState.generateSuccessor(agentIndex, a),
                          gameState.getLegalActions(agentIndex))

    if self.gameOver(gameState, depth):
      return self.evaluationFunction(gameState)

    else:
      if agentIndex == 0: # pacman
        v = float("inf") #alpha beta was weird without doing the infinity part of the algorithm
        for successor in successorStates:
          v = max(self.alphabeta((agentIndex + 1) % gameState.getNumAgents(), 
                                 successor, depth-1, alpha, beta), v)
          if v >= beta:
            return v
          alpha = max(alpha, v)
        return v

      else: # ghost
        v = float("inf")
        for successor in successorStates:
          v = min(self.alphabeta((agentIndex + 1) % gameState.getNumAgents(),
                                 successor, depth-1, alpha, beta), v)
          if v <= alpha:
            return v
          beta = min(beta, v)
        return v

  def getAction(self, gameState):
    depth = gameState.getNumAgents() * self.depth

    legalActions = gameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)

    successorStates = map(lambda a: gameState.generateSuccessor(0,a),
                          legalActions)

    vals = map(lambda s: self.alphabeta(1, s, depth - 1, -1e308, 1e308), 
               successorStates)

    return legalActions[vals.index(max(vals))]


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
    
  def gameOver(self, gameState, d):
    return gameState.isLose() or gameState.isWin() or d == 0

  def expectimax(self, gameState, agentIndex,  depth):
    """
    Same as minimax, except we do an average of min. 
    We do an average because the ghost behavior is expected to be 
    'uniformly at random'. If that's the case, then the expected
    value of a node's children is the average of their values.
    """
    successorStates = map(lambda a: gameState.generateSuccessor(agentIndex, a),
                         gameState.getLegalActions(agentIndex))

    if self.gameOver(gameState, depth): # at an end
      return self.evaluationFunction(gameState)
    else:

      newIndex = (agentIndex + 1) % gameState.getNumAgents()
      vals = map(lambda s: self.expectimax(s, newIndex, depth - 1), 
                successorStates)      

      if agentIndex == 0: # pacman
        return max(vals)
      else: # ghost, here's the expectimax part.
        return sum(vals)/len(vals)
        
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    depth = gameState.getNumAgents() * self.depth

    legalActions = gameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)

    successorStates = map(lambda a: gameState.generateSuccessor(0,a),
                         legalActions)
    vals = map(lambda s: self.expectimax(s, 1, depth - 1), 
    successorStates)
    return legalActions[vals.index(max(vals))]

def testEval(currentGameState):
  
  pos = currentGameState.getPacmanPosition()
  currentScore = scoreEvaluationFunction(currentGameState)


  if currentGameState.isLose(): 
    return -float("inf")
  elif currentGameState.isWin():
    return float("inf")

  # food distance
  foodlist = currentGameState.getFood().asList()
  manhattanDistanceToClosestFood = min(map(lambda x: util.manhattanDistance(pos, x), foodlist))
  distanceToClosestFood = manhattanDistanceToClosestFood

  # number of big dots
  numberOfCapsulesLeft = len(currentGameState.getCapsules())
  
  # number of foods left
  numberOfFoodsLeft = len(foodlist)
  
  # ghost distance

  # active ghosts are ghosts that aren't scared.
  scaredGhosts, activeGhosts = [], []
  for ghost in currentGameState.getGhostStates():
    if not ghost.scaredTimer:
      activeGhosts.append(ghost)
    else: 
      scaredGhosts.append(ghost)

  def getManhattanDistances(ghosts): 
    return map(lambda g: util.manhattanDistance(pos, g.getPosition()), ghosts)

  distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0

  if activeGhosts:
    distanceToClosestActiveGhost = min(getManhattanDistances(activeGhosts))
  else: 
    distanceToClosestActiveGhost = float("inf")
  distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)
    
  if scaredGhosts:
    distanceToClosestScaredGhost = min(getManhattanDistances(scaredGhosts))
  else:
    distanceToClosestScaredGhost = 0 # I don't want it to count if there aren't any scared ghosts

  outputTable = [["dist to closest food", -1.5*distanceToClosestFood], 
                 ["dist to closest active ghost", 2*(1./distanceToClosestActiveGhost)],
                 ["dist to closest scared ghost", 2*distanceToClosestScaredGhost],
                 ["number of capsules left", -3.5*numberOfCapsulesLeft],
                 ["number of total foods left", 2*(1./numberOfFoodsLeft)]]

  # a, b, c, d, e, and f are all constants given through the command line.
  # they're set/declared as global variables in a method at the top of this file
  # and through the expectimax constructor.
  #print _a, ", ", _b, ", ", _c, ", ", _d, ", ", _e, ", ", _f
  # print(type(float(_a)))
  # print(type(float(_b)))
  # print(type(float(_c)))
  # print(type(float(_d)))
  # print(type(_e))
  # print(type(float(_f)))

  score = _a   * currentScore + \
          _b   * distanceToClosestFood + \
          _c   * (1./distanceToClosestActiveGhost) + \
          _d   * distanceToClosestScaredGhost + \
          _e   * numberOfCapsulesLeft + \
          _f   * numberOfFoodsLeft

  # score = 1    * currentScore + \
  #         -1.5 * distanceToClosestFood + \
  #         2    * (1./distanceToClosestActiveGhost) + \
  #         2    * distanceToClosestScaredGhost + \
  #         -3.5 * numberOfCapsulesLeft + \
  #         -4    * numberOfFoodsLeft
  return score


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    
    DESCRIPTION: <write something here so we know what you did>
  """
  return testEval(currentGameState)

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
    
  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.
      
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
