import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np

# Define the graph schema
graph_schema = tfgnn.GraphSchema(
    node_sets={"nodes": tfgnn.NodeSet()}  # Define the node set
)

# Create a graph tensor
node_features = {"nodes": {"features": tf.constant(np.random.rand(10, 5), dtype=tf.float32)}}
graph_tensor = tfgnn.create_graph_tensor(graph_schema, node_features)

# Define a simple Graph Convolutional Layer
class SimpleGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape[-1], self.units])

    def call(self, graph_tensor):
        node_features = graph_tensor.node_sets["nodes"]["features"]
        outputs = tf.matmul(node_features, self.kernel)
        return graph_tensor.replace_features(node_sets={"nodes": {"features": outputs}})

# Create the GCN model
inputs = tf.keras.layers.Input(type_spec=graph_tensor.spec)
outputs = SimpleGCNLayer(units=16)(inputs)
model = tf.keras.Model(inputs, outputs)

# Compile and train the model (example only)
model.compile(optimizer='adam', loss='mse')
