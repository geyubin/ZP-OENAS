from Operation import Operations_4_name

class dagnode:

    def __init__(self, node_id, adj_node, op_id):
        self.node_id = node_id
        self.adj_node = adj_node
        self.op_id = op_id
        self.op_name = Operations_4_name[op_id]

