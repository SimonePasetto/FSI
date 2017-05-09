# Search methods

import search
import sys

ab = search.GPSProblem('A', 'B', search.romania)


sys.stdout.write("Ramificacion y acotacion -- Nodos expandido: ")
print search.ramificacion_y_acotacion(ab).path()
sys.stdout.write("Ramificacion y acotacion con subestimacion -- Nodos expandido: ")
print search.ramificacion_y_acotacion_con_subestimacion(ab).path()
sys.stdout.write("Busqueda en enchura -- Nodos expandido: ")
print search.breadth_first_graph_search(ab).path()
sys.stdout.write("Busqueda en profundidad -- Nodos expandido: ")
print search.depth_first_graph_search(ab).path()
# print search.iterative_deepening_search(ab).path()
# print search.depth_limited_search(ab).path()

#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
