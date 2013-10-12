# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]


def search(problem, frontier, heuristic, getSuccessorFunction):
  """
  search is an abstracted search function that can perform dfs, bfs, ucs, and A*
  The difference between the algorithms is the data structure used for the 
  frontier, and the heuristic used.
  Data Structures:
  - Priority Queue: informed search 
  - Stack: Depth First Search
  - Queue: Breadth First Search

  If the heuristic is None, it will be uninformed. If the heuristic is the 
  nullHeuristic, it will perform Uniform Cost Search. Otherwise, it will
  perform A* search using the given heuristic.
  """
  # g, h, and f are hash tables where the keys are states (coordinates), and the
  # the value of some key k is k's value for the respective function.
  g = {}  # the cost to get to the key node
  f = {}  # the estimated cost from the key node to the goal
  h = {}  # h[k] = f[k] + g[k]

  visited = {}
  path = []
  start = problem.getStartState()

  # each node is represented as a tuple:
  # - (Current, Parent, Direction)
  # where Current and Parent are problems, and Direction
  # is the direction taken from Parent to get to Current
  # each node in the frontier is represented in this way.

  if heuristic: 
    g[start] = 0
    h[start] = heuristic(start, problem)
    f[start] = g[start] + h[start]
    frontier.push((start, start, False), f[start])
  else: 
    frontier.push((start, start, False))

  def push(node, parent):
    "push the node/parent onto the frontier"
    if heuristic: # if we have a heuristic      
      frontier.push((node[0], parent[0], node[1]), f[node[0]])
    else:
      frontier.push((node[0], parent[0], node[1]))

  while not(frontier.isEmpty()):
    currentNode = frontier.pop()

    def distanceFromState(state, point):
      "return manhattan distance from state to point"
      pos = state[1]
      return abs(pos[0] - point[0]) + abs(pos[1] - point[1])


    # if we've found the goal, iterate to the root and construct the path
    if problem.isGoalState(currentNode[0]):
      while currentNode[2]:
        path.insert(0, currentNode[2])
        currentNode = visited[currentNode[1]]
      return path

    visited[currentNode[0]] = currentNode

    nonVisited = [successor for successor 
                  in problem.getSuccessors(currentNode[0]) 
                  if successor[0] not in visited]

    for successor in nonVisited:
      
      if heuristic: # if we have a heuristic
        def updatePath(n):
          "update the path and cost functions"
          g[n[0]] = tentativeGScore
          f[n[0]] = g[n[0]] + h[n[0]]
          push(n, currentNode)

        tentativeGScore = g[currentNode[0]] + successor[2]

        if not h.has_key(successor[0]): # we don't have a heuristic value yet
          h[successor[0]] = heuristic(successor[0], problem)
          updatePath(successor)
        elif tentativeGScore < g[successor[0]]: # we found a better score
          updatePath(successor)

      else: # we're uninformed
        push(successor, currentNode)

  return None


def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  """
  return search(problem, util.Stack(), None)


def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  return search(problem, util.Queue(), None)

    
        
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  return search(problem, util.PriorityQueue(), nullHeuristic)


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  """
  Search the node that has the lowest combined cost and heuristic first.
  """
  return search(problem, util.PriorityQueue(), heuristic)

  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
