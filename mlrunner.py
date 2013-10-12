#!/usr/bin/env python
import subprocess, re, random

# this script generates a training set of pacman runs. 
# it randomly generates a coefficient between -4 and 4 
# for each feature. 

destfile = "/home/dan/classes/ai/training.txt"
def runPacman(c):
    "runs pacman with the given constants, returns the score #"
    # the coefficients are sent in as additional arguments to the eval
    # function
    betterArg = "evalFn=better,a={},b={},c={},d={},e={},f={}".format(c[0], c[1], c[2], c[3], c[4], c[5])
    output = subprocess.check_output(["python", "pacman.py", "-l", "smallClassic", "-p", "ExpectimaxAgent", "-a", betterArg, "-n", "10", "-q"])
    # regex = "Pacman emerges victorious! Score: (.+?)\n "
    # winningScores =  map(int,re.findall(regex, output))
    # if winningScores:
    #     averageWinningScore = sum(winningScores)/len(winningScores)
    # else:
    #     averageWinningScore = 0
    # return averageWinningScore
    regex = "Average Score: (.+?)\n"
    averageScore = float(re.findall(regex, output)[0])
    return averageScore

def saveScore(constants, score):
    line = str(tuple(constants)).strip(")").strip("(") + " : " + str(score) + "\n"
    with open(destfile, 'a') as outfile:
        outfile.write(line)

constants = [1, 0, 0, 0, 0, 0]

# score = runPacman(constants)
# saveScore(constants, score)

#runPacman(constants)
i = 0
while i < 1000:
    a = random.uniform(-4.0, 4.0)
    b = random.uniform(-4.0, 4.0)
    c = random.uniform(-4.0, 4.0)
    d = random.uniform(-4.0, 4.0)
    e = random.uniform(-4.0, 4.0)
    f = random.uniform(-4.0, 4.0)
    constants = [a, b, c, d, e, f]
    score = runPacman(constants)
    saveScore(constants, score)    



# python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better,a=1,b=2, -n 1 -q
