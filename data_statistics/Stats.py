class Stats:
    def __init__(self):
        """ """
        self.data = {
            "mlp": {
                ("frog", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_temp", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_gauss", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("no_frog", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_temp", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_gauss", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("no_frog", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
            },
            "logistic_regression": {
                ("frog", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_temp", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_gauss", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("no_frog", "clean"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_temp", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("frog_gauss", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
                ("no_frog", "dirty"): {
                    "validation": {
                        "accuracy": [],
                        "roc_auc": [],
                        "balanced_accuracy": [],
                        "f1": [],
                        "loss": [],
                    },
                    "test": {
                        "accuracy": 0,
                        "roc_auc": 0,
                        "balanced_accuracy": 0,
                        "f1": 0,
                        "loss": 0,
                    },
                },
            },
        }

    # Generic Eval Metric
    def set_eval_metric(self, model, split, algorithm, dq_type, metric, value):
        if split == "validation":
            self.data[model][split][(algorithm, dq_type)][metric].append(value)
        else:
            self.data[model][split][(algorithm, dq_type)][metric] = value

    def get_eval_metric(self, model, split, algorithm, dq_type, metric):
        return self.data[model][split][(algorithm, dq_type)][metric]

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
