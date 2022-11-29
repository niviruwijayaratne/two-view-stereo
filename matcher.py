import numpy as np

class Matcher:
    def __init__(self, matching_method, window_size):
        self.match = None
        self.wsize = window_size
        if matching_method == "SSD":
            self.match = self.ssd
        elif matching_method == "NCC":
            self.match = self.ncc
        else:
            print("Matching method not currently implemented")
    
    def ssd(self, window1, window2):
        matching_costs = np.sum(np.square(window1 - window2), axis=1)
        return matching_costs
    
    def sad(self, window1, window2):
        matching_costs = np.sum(np.abs(window1 - window2), axis=1)
        return matching_costs

    def ncc(self, window1, window2):
        matching_cost_num = np.sum(np.multiply(window1, window2), axis=1)
        matching_cost_denom = np.sqrt(np.sum(np.square(window1)) * np.sum(np.square(window2), axis=1))
        matching_costs = matching_cost_num/matching_cost_denom
        return matching_costs