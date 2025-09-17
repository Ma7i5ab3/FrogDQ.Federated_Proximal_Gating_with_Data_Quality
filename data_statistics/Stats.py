class Stats:
    def __init__(self):
        """
        Parameters:

        topk_features: (list); it is used to track the updates of g and w over the rounds. Only the features with the highest absolute value in the model will be tracked.
        data_type: (str); it defines if the instance tracks experiment with 'clean' or 'dirty' data. Possible values: 'dirty' or 'clean'
        method: (str); it defines the federated learning algorithm used. Possible values: 'fedavg', 'fedprox' or 'frog'
        """
        self.topk_features = []
        self.data = {
            ("fedavg", "clean"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("fedprox", "clean"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("frog", "clean"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("frog_new", "clean"): {
               "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("fedavg", "dirty"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("fedprox", "dirty"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("frog", "dirty"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("frog_new", "dirty"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            },
            ("fedavg_no_corr_feat", "dirty"): {
                "accuracy_test": [],
                "roc_auc_test": [],
                "accuracy_val": [],
                "roc_auc_val": [],
                "loss_val": [], 
                "loss": [],
                "gate_updates": [],
                "weight_updates": [],
            }
        }

    # Topk_features
    def set_topk_features(self, value):
        self.topk_features.append(value)

    def get_topk_features(self):
        return self.topk_features

    # Generic Eval Metric
    def set_eval_metric(self, algorithm, dq_type, metric, value):
        self.data[(algorithm, dq_type)][metric].append(value)
    
    def get_eval_metric(self, algorithm, dq_type, metric):
        return self.data[(algorithm, dq_type)][metric]

    # Accuracy test
    def set_accuracy_test(self, algorithm, dq_type, value):
        self.data[(algorithm, dq_type)]["accuracy_test"].append(value)

    def get_accuracy_test(self, algorithm, dq_type):
        return self.data[(algorithm, dq_type)]["accuracy_test"]
    
    # Accuracy val
    def set_accuracy_val(self, algorithm, dq_type, value):
        self.data[(algorithm, dq_type)]["accuracy_val"].append(value)

    def get_accuracy_val(self, algorithm, dq_type):
        return self.data[(algorithm, dq_type)]["accuracy_val"]

    # Loss
    def set_val_loss(self, algorithm, dq_type, value):
        self.data[(algorithm, dq_type)]["loss_val"].append(value)

    def get_val_loss(self, algorithm, dq_type):
        return self.data[(algorithm, dq_type)]["loss_val"]

    # Gate updates
    def set_gate_updates(self, algorithm, dq_type, value):
        self.data[(algorithm, dq_type)]["gate_updates"].append(value)

    def get_gate_updates(self, algorithm, dq_type):
        return self.data[(algorithm, dq_type)]["gate_updates"]

    # Weights
    def set_weights_updates(self, algorithm, dq_type, value):
        self.data[(algorithm, dq_type)]["weight_updates"].append(value)

    def get_weights_updates(self, algorithm, dq_type):
        return self.data[(algorithm, dq_type)]["weight_updates"]

    # Reset all stats
    def reset(self, stats_field: str = None):
        if stats_field is None:
            self.topk_features.clear()
            for key in self.data:
                for stat in self.data[key]:
                    self.data[key][stat].clear()
        else:
            for key in self.data:
                self.data[key][stats_field].clear()
