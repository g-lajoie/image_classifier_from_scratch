from .base_layers import Layer


class LayerStack:
    """
    Prepares sequential layers for Neural Network.

    Attributes:
        *layers: Tuple of layers.
    """

    def __init__(self, *layers: Layer):
        self.layers = layers

        # Initialization Methods
        self.set_layer_hierarchy()

    def set_layer_hierarchy(self):
        """
        Helper function that set next layer attribute of each layer.
        """

        for i in range(len(self.layers)):
            if i != 0:
                self.layers[i].parent_layer = self.layers[i - 1]

            if i != len(self.layers) - 1:
                self.layers[i].child_layer = self.layers[i + 1]
