

class RegionPredictor:
    """
    Responsible for predicting Terra Regions according to task
    """
    
    def __init__(self, terra):
        self.terra = terra
        
    def predict(self, tasks_tensor, task_names, method="ms"):
        pass