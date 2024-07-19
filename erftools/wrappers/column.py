from .wrapper import ABLWrapper

class GeostrophicWindEstimator(ABLWrapper):
    """This will estimate the geostrophic wind that gives a specified
    wind speed at a reference height
    """

    def __init__(self):
        super().__init__()
