import random
import heapq
from collections import defaultdict
random.seed(42)
import math
def create_random_matrix(n):
    matrix = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    return matrix

n = 10 
random_matrix = create_random_matrix(n)

for row in random_matrix:
    print(row)


def four_nei(node):
    x, y = node
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]




class Diastraj:
    def __init__(self):

        self.visited = set()

    def diastraj(self, matrix, start, end):
        minqueue = []

        x_bound, y_bound = len(matrix), len(matrix[0])
        x1, y1 = start

        xe, ye= end
        heapq.heappush(minqueue, (matrix[x1][y1], start))

        while minqueue:

            dis, node = heapq.heappop(minqueue)

            self.visited.add(node)

            if node == end:
                return dis
            
            neis = four_nei(node)

            for nei in neis:

                i, j = nei
                if 0 <= i < x_bound and 0 <= j < y_bound and nei not in self.visited:
                    
                    heapq.heappush(minqueue, (dis+matrix[i][j], nei))


Diastraj_obj = Diastraj()
Diastraj_dis = Diastraj_obj.diastraj(random_matrix, (0,0), (len(random_matrix)-1, len(random_matrix[0])-1))              
print(Diastraj_dis)

steps = sorted(list(Diastraj_obj.visited))
# print(steps)
print(f"number is steps is {len(steps)}")
        


class A_star:

    def __init__(self):

        self.visited = set()
        self.h = defaultdict(lambda: float('inf'))
        self.g = defaultdict(lambda: float('inf'))
        self.open = set()

    def A_star(self, matrix, start, end):

        visited = set()
        minqueue = []

        x_bound, y_bound = len(matrix), len(matrix[0])
        x1, y1 = start

        xe, ye= end

        # l2_dis = abs(xe-x1) + abs(ye-y1)
        l2_dis = math.sqrt((xe-x1)**2 + (ye-y1)**2)
        self.g[start] = matrix[x1][y1]

        heapq.heappush(minqueue, (l2_dis, matrix[x1][y1], start))
        self.open.add(start)
        while minqueue: 
            
            dis_heuris, dis, node = heapq.heappop(minqueue)

            if node == end:
                return dis
            
            neis = four_nei(node)

            for nei in neis:

                i, j = nei
                if 0 <= i < x_bound and 0 <= j < y_bound:
                    
                    curr_g = dis + matrix[i][j]
                    # l2_dis = abs(xe-i) + abs(ye-j)
                    l2_dis = math.sqrt((xe-i)**2 + (ye-j)**2)
                    if nei in self.open:
                        if curr_g >= self.g[nei]:
                            continue

                    elif nei in self.visited:
                        if curr_g >= self.g[nei]:
                            continue
                        self.visited.remove(nei)
                        self.open.add(nei)
                        heapq.heappush(minqueue, (curr_g+l2_dis, curr_g, nei))

                    else:
                        self.open.add(nei)
                        heapq.heappush(minqueue, (curr_g+l2_dis, curr_g, nei))

                    self.g[nei] = curr_g
            self.visited.add(node)

A_star_obj = A_star()
A_star_dis = A_star_obj.A_star(random_matrix, (0,0), (len(random_matrix)-1, len(random_matrix[0])-1))
steps = sorted(list(A_star_obj.visited))
# print(steps)
print(A_star_dis)
print(f"number is steps is {len(steps)}")


class A_star_2:
    def __init__(self):
        self.visited = set()
        self.came_from = {}
        self.h = defaultdict(lambda: float('inf'))
        self.g = defaultdict(lambda: float('inf'))
        self.open = []
        heapq.heapify(self.open)

    def A_star(self, matrix, start, end):
        def heuristic(node):
            x1, y1 = node
            x2, y2 = end
            return abs(x1 - x2) + abs(y1 - y2)

        self.h[start] = heuristic(start)
        x1, y1 = start
        self.g[start] = matrix[x1][y1]
        heapq.heappush(self.open, (self.h[start], start))

        while self.open:
            current_cost, current_node = heapq.heappop(self.open)

            if current_node == end:
                path = [current_node]
                while current_node != start:
                    current_node = self.came_from[current_node]
                    path.append(current_node)
                return current_cost

            self.visited.add(current_node)

            for neighbor in four_nei(current_node):
                if neighbor[0] >= 0 and neighbor[0] < n and neighbor[1] >= 0 and neighbor[1] < n:
                    tentative_g = self.g[current_node] + matrix[neighbor[0]][neighbor[1]]

                    if tentative_g < self.g[neighbor]:
                        self.came_from[neighbor] = current_node
                        self.g[neighbor] = tentative_g
                        self.h[neighbor] = tentative_g + heuristic(neighbor)
                        heapq.heappush(self.open, (self.h[neighbor], neighbor))

        return None
    
A_star_obj = A_star_2()
A_star_dis = A_star_obj.A_star(random_matrix, (0,0), (len(random_matrix)-1, len(random_matrix[0])-1))
steps = sorted(list(A_star_obj.visited))
# print(steps)
print(A_star_dis)
print(f"number is steps is {len(steps)}")
