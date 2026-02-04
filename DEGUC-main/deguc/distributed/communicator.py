class DistributedCommunicatorPlaceholder:
    """
    this is a stock
    """
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank

    def dispatch_tokens(self, hidden, routing):
        return hidden, routing

    def collect_outputs(self, outputs):
        return outputs
