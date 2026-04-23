import pickle


class Results:
    def __init__(self, flux_data, labeling_data, loss_log, model):
        self.flux_data = flux_data
        self.labeling_data = labeling_data
        self.loss_log = loss_log
        self.model = model

    def save_results(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_results(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
