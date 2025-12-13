"""
Custom evaluator for non-OGB datasets.

Mimics the OGB Evaluator interface with simple accuracy metric.
"""


class CustomEvaluator:
    """
    Evaluator for custom datasets, mimicking OGB Evaluator interface.

    Usage:
        evaluator = CustomEvaluator()
        result = evaluator.eval({
            'y_true': labels,      # [N, 1] tensor
            'y_pred': predictions  # [N, 1] tensor
        })
        accuracy = result['acc']
    """

    def eval(self, input_dict):
        """
        Evaluate predictions against ground truth.

        Args:
            input_dict: Dictionary with 'y_true' and 'y_pred' tensors

        Returns:
            Dictionary with 'acc' key containing accuracy value
        """
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]

        # Handle different tensor shapes
        if y_true.dim() > 1:
            y_true = y_true.squeeze()
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()

        correct = (y_true == y_pred).sum().item()
        total = y_true.shape[0]

        return {"acc": correct / total if total > 0 else 0.0}
