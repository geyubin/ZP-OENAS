import numpy as np


def compare_(fitness_curr, fitness_ref):
    for i in range(len(fitness_curr)):
        if fitness_curr[i] < fitness_ref[i]:
            return True
    return False


def judge_(fitness_curr, fitness_data, cursor):
    
    for i in range(len(fitness_data)):
        if i == cursor:
            continue
        
        if not compare_(fitness_curr, fitness_data[i]):
            return False
    return True


class Pareto_:
    def __init__(self, in_data, fitness_data):
        self.in_data = in_data  
        self.fitness_data = fitness_data  
        self.cursor = -1  
        self.len_ = len(in_data)  
        self.bad_num = 0 

    def next(self):
       
        self.cursor = self.cursor + 1
        return self.in_data[self.cursor], self.fitness_data[self.cursor]

    def hasNext(self):
   
        return self.len_ > self.cursor + 1 + self.bad_num

    def remove(self):
    
        self.fitness_data = np.delete(self.fitness_data, self.cursor, axis=0)
        self.in_data = np.delete(self.in_data, self.cursor, axis=0)
        
        self.cursor = self.cursor - 1
     
        self.bad_num = self.bad_num + 1

    def pareto(self):
        while self.hasNext():

            in_curr, fitness_curr = self.next()

            if not judge_(fitness_curr, self.fitness_data, self.cursor):
                self.remove()
        return self.in_data, self.fitness_data
