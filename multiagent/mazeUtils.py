def getSuccessorsInMaze(pos, walls):
    "returns a list of successors limited by the design of the maze"

    def notInWall(pos):
        "returns true if the position is not in a wall"
        x = pos[0]
        y = pos[1]

        # true if the position is in the boundaries of the maze
        inBounds =  not (x <= 0 or 
                         x >= walls.width or 
                         y <= 0 or 
                         y >= walls.height)

        # returns true if the posn isn't in a wall or out of bounds
        return (not walls[pos[0]][pos[1]] ) and inBounds
        
    def getSuccessors():
        "returns a list of all possible successors"
        x = pos[0]
        y = pos[1]
        successors = []
        successors.append((x+1, y+1))
        successors.append((x+1, y))
        successors.append((x+1, y-1))
        successors.append((x, y-1))
        successors.append((x-1, y-1))
        successors.append((x-1, y))
        successors.append((x-1, y+1))
        successors.append((x, y+1))
        return successors

    return filter(notInWall, getSuccessors())


def getSuccessorsForBFS(position, walls):
  successors = []
  x = position[0]
  y = position[1]
  
  if(x - 1 >= 0):
    if(not walls[x - 1][y]):
      successors.append((x - 1, y))
    if(y - 1 >= 0):
        if(not walls[x - 1][y - 1]):
          successors.append((x - 1, y - 1))
    if(y + 1 < walls.height):
      if(not walls[x - 1][y + 1]):
        successors.append((x - 1, y + 1))
        
  if(x + 1 < walls.width):
    if(not walls[x + 1][y]):
      successors.append((x + 1, y))
    if(y - 1 >= 0):
        if(not walls[x + 1][y - 1]):
          successors.append((x + 1, y - 1))
    if(y + 1 < walls.height):
      if(not walls[x + 1][y + 1]):
        successors.append((x + 1, y + 1))
      
    if(y - 1 >= 0):
      if(not walls[x][y - 1]):
        successors.append((x, y - 1))
        
    if(y + 1 < walls.height):
      if(not walls[x][y + 1]):
        successors.append((x, y + 1))
  
  return successors

def distanceInMaze(start, goal, walls):
    "bfs to find distance from start to goal in the maze"
    import util
    frontier = util.Queue()
    visited = []
    #print "pushing: ", start, " and ", 0
    frontier.push((start, 0)) # each node is a tuple: (pos, dist)
    
    while not frontier.isEmpty():
        currentNode = frontier.pop()
        #print currentNode
        if currentNode[0] == goal:
            return currentNode[1]
        nonVisited = [successor for successor in 
                      getSuccessorsInMaze(currentNode[0], walls) 
                      if successor not in visited]
        for successor in nonVisited:
            visited.append(successor)
            frontier.push((successor, 1 + currentNode[1]))
    return None

_walls = None

def distancesInMaze(walls, distanceDict):
    """
    Returns a hash of (start, goal) -> distance
    where start and goal are both posns, and distance is the
    'maze-distance' between the start and goal.
    """
    posns = []
    for x in range(walls.width):
        for y in range(walls.height): 
            posns.append((x, y))

    for start in posns:
        for goal in posns:
            sg = (start, goal)
            gs = (goal, start)
            if (sg not in distanceDict and 
                gs not in distanceDict and
                not walls[start[0]][start[1]] and
                not walls[goal[0]][goal[1]]):
         #       print distanceInMaze(start, goal, walls)
                distanceDict[sg] = distanceInMaze(start, goal, walls)
