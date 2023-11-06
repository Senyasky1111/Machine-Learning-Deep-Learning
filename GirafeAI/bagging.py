import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        self.indices_list = []
        self.models_list = []
        self.data = None
        self.target = None
        self.oob_predictions = None

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = [np.random.choice(len(data), len(data), replace=True) for _ in range(self.num_bags)]

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = data
        self.target = target
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain len(data) number of elements!'
        
        for bag_indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[bag_indices], target[bag_indices]
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = np.array([model.predict(data) for model in self.models_list])
        return np.mean(predictions, axis=0)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during the training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]

        for i, instance in enumerate(self.data):
            for j, indices in enumerate(self.indices_list):
                if i not in indices:
                    list_of_predictions_lists[i].append(self.models_list[j].predict([instance])[0])

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from the training set.
        If an object has been used in all bags during the training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([np.mean(predictions) if predictions else None for predictions in self.list_of_predictions_lists])

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        
        valid_indices = [i for i, predictions in enumerate(self.list_of_predictions_lists) if predictions]
        squared_errors = [(self.target[i] - self.oob_predictions[i])**2 for i in valid_indices]
        mse = sum(squared_errors) / len(valid_indices) if valid_indices else None
        return mse