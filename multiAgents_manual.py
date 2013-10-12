from game import Directions, Agent
import random, util

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

    # getting closer to food is good
    # getting closer to ghosts is bad

    foodScore = 0
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
  
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.depth = int(depth)
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

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    
  I used the following features in this model:
  - distance to the closest active ghost (active ghosts are non-scared ghosts)
  - current score in the game
  - distance to the closest scared ghost
  - number of capsules left
  - number of foods left
  - distance to the closest food

  My evaluation function computes a linear combination of 
  these features (or related features, since in some cases, I take
  the inverse of a feature)

  I kept the current score the same, because I saw no reason to modify it.

  I multiply the distance to the closest food by -1.5. This means that the 
  larger the distance pac-man has to the closest food, the more negative the
  score is.

  I take the inverse of the distance to the closest active ghost, and then 
  multiply it by -2. This means that the larger the distance to the closest
  active ghost, the les negative the score is, but the closer a ghost is, 
  the more negative the score becomes.

  I multiply distance to the closest scared ghost by -2, to motivate pac-man to 
  move towards scared ghosts. This coefficient is larger than the coefficient I
  used for food, even though the distance to the closest scared ghost will
  almost always be greater than the distance to the nearest food. I chose to
  use a larger coefficient here because:
   - pac-man gets a large number of points for eating a scared ghost
   - if the distance to the closest scared ghost is greater than the distance
     to the nearest food, it is usually because there are not many foods left
     on the board. This means that it's likely more beneficial for pac-man to go
     towards the scared ghost, because eating the scared ghost will likely net 
     pac-man more points than eating the remaining foods.

  I multiply the number of capsules left by a very high negative number - -20 - 
  in order to motivate pac-man to eat capsules that he passes. I didn't want
  pac-man to move toward capsules over food or over running away from ghosts, 
  but I DID want pac-man to eat them when he passed by them. When pac-man 
  passes by a capsule, the successor state where pac-man eats a capsule will
  gain +20 points, which is (usually) significant enough that pac-man eats 
  the capsule.

  I also multiply the number of foods left by -4, because pac-man should
  minimize the number of foods that are left on the board. 

  some results: 

  dan at staircar in ~/classes/ai/multiagent on master
  $ python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -n 10 -q
  Pacman emerges victorious! Score: 1366
  Pacman died! Score: -198
  Pacman emerges victorious! Score: 1367
  Pacman emerges victorious! Score: 1737
  Pacman emerges victorious! Score: 1364
  Pacman emerges victorious! Score: 933
  Pacman emerges victorious! Score: 1743
  Pacman emerges victorious! Score: 1193
  Pacman emerges victorious! Score: 1373
  Pacman emerges victorious! Score: 1348
  Average Score: 1222.6
  Scores:        1366, -198, 1367, 1737, 1364, 933, 1743, 1193, 1373, 1348
  Win Rate:      9/10 (0.90)
    Record:        Win, Loss, Win, Win, Win, Win, Win, Win, Win, Win


  dan at staircar in ~/classes/ai/multiagent on master
  $ python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -n 10
  Pacman emerges victorious! Score: 1139
  Pacman emerges victorious! Score: 1362
  Pacman emerges victorious! Score: 1770
  Pacman emerges victorious! Score: 1361
  Pacman emerges victorious! Score: 1234
  Pacman emerges victorious! Score: 1521
  Pacman emerges victorious! Score: 1755
  Pacman emerges victorious! Score: 1759
  Pacman emerges victorious! Score: 1759
  Pacman died! Score: 101
  Average Score: 1376.1
  Scores:        1139, 1362, 1770, 1361, 1234, 1521, 1755, 1759, 1759, 101
  Win Rate:      9/10 (0.90)
  Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Loss

  ---
  I also experimented with using the actual maze-distance instead of the
  manhattan distance, but that turned out to be mostly useless. 

  I spent a significant amount of time trying to develop a machine-learning
  solution to this problem by generating a training set that I could run linear
  regression on. I had difficult generating a good training set, but I think that
  this problem could be solved easily using that solution with the right data. 

  The script I used/played around with to create the training data is included. 
   
  """
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
  # if we only count the number fo them, he'll only care about
  # them if he has the opportunity to eat one.
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

  score = 1    * currentScore + \
          -1.5 * distanceToClosestFood + \
          -2    * (1./distanceToClosestActiveGhost) + \
          -2    * distanceToClosestScaredGhost + \
          -20 * numberOfCapsulesLeft + \
          -4    * numberOfFoodsLeft
  return score

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
