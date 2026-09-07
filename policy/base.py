from abc import ABC, abstractmethod
from params import Params
from common.customTypes import NodeDecision
from metrics import Metrics

class NodeDecisionPolicy(ABC):
    def __init__(self, params: Params):
        self.params = params
        self.initialized = False

    @abstractmethod
    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        pass

    def IsInitialized(self) -> bool:
        return self.initialized

class NoDecisionPolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        super().__init__(params)
        self.initialized = True

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        # return NodeDecision.MAP
        return NodeDecision.LOCALIZE