# Created by Hugo Nolte for course PA1414 - DoBot Magician Project
# 2019

import threading
import DoBotArm as Dbt

#Example of bundling functions
def functons():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ) #Create DoBot Class Object with home position x,y,z
    ctrlBot.moveArmXY(250, 100)
    ctrlBot.pickToggle(-40)
    ctrlBot.toggleSuction()
    ctrlBot.pickToggle(-40)
    ctrlBot.moveHome()
    ctrlBot.pickToggle(-40)
    ctrlBot.toggleSuction()
    ctrlBot.pickToggle(-40)

#An example combining the functions into a manual control mode
def manualMode():
    homeX, homeY, homeZ = 250, 0, 50
    ctrlBot = Dbt.DoBotArm(homeX, homeY, homeZ) #Create DoBot Class Object with home position x,y,z

    print("---Manual Mode---")
    print("move to move to location")
    print("pick - toggles picking at certain height")
    print("suct - toggles suction on and off")
    print("q - exit manual mode")
    while True:
        inputCoords = input("$ ")
        inputCoords = inputCoords.split(",")
        if(inputCoords[0] == "move"):
            x = int(inputCoords[1])
            y = int(inputCoords[2])
            ctrlBot.moveArmXY(x,y)
        elif(inputCoords[0] == "pick"):
            height = int(inputCoords[1])
            ctrlBot.pickToggle(height)
        elif(inputCoords[0] == "suct"):
            ctrlBot.toggleSuction()
        elif(inputCoords[0] == "q"):
            break
        else:
            print("Unrecognized command")


#--Main Program--
def main():
    manualMode()

main()
