import torch
import torch.nn as nn
from transformers import Trainer
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Union,
)

# https://github.com/huggingface/transformers/issues/7232

"""
    When the development set is too big and it does not fit in the GPU memory, we have
    to perform the evaluation during the training using the CPU. This custom trainer allows that.
    See parameter eval_device.
"""

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_device=torch.device('cpu'), **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_device = eval_device

    def prediction_step(
        self, model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            #loss = loss.mean().item()

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        
        # move tensors to evaluation device
        ret = (loss, logits.detach().to(self.eval_device), labels.to(self.eval_device))
        return ret