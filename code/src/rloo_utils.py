import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader

from accelerate import Accelerator

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

from accelerate.utils import is_deepspeed_available
from transformers import PreTrainedModel, PreTrainedTokenizer

import itertools
import numpy as np


if is_deepspeed_available():
    import deepspeed

if TYPE_CHECKING:
    from accelerate import Accelerator
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch.nn.parallel.distributed import DistributedDataParallel



class CustomDataset(Dataset):
    def __init__(self, data_file_path):
        with open(data_file_path, "r") as f:
            lines = [json.loads(l) for l in f.readlines()]
        
        self.raw_data = lines
            
    def __getitem__(self, index):
        
        return self.raw_data[index]
    
    def __len__(self):
        
        return len(self.raw_data)


class ExpMinDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
            
    def __getitem__(self, index):
        
        
        try:
            return  {key: self.data_dict[key][index] for key in self.data_dict}
        except:
            
            return_dict =  {key: self.data_dict[key][0] for key in self.data_dict}
            return_dict["weights"] = torch.tensor([0])
            return return_dict
    
    def __len__(self):
        
        if "input_ids" not in self.data_dict:
            return 0
        
        return len(self.data_dict["input_ids"])
    
    def get_data_dict(self):
        return self.data_dict

    def add(self, new_dataset):
        data_dict = new_dataset.get_data_dict()
        for key in data_dict.keys():
            try:
                if key in self.data_dict:
                    self.data_dict[key] = torch.cat([self.data_dict[key], data_dict[key]], dim=0)
                else:
                    self.data_dict[key] = data_dict[key]
            except:
                print(f"Error in key {key}")
                print(f"self.data_dict[key].shape: {self.data_dict[key].shape}")
                print(f"data_dict[key].shape: {data_dict[key].shape}")
                
                
            
    
    def shuffle(self):
        perm = torch.randperm(len(self.data_dict["input_ids"]))
        for key in self.data_dict:
            self.data_dict[key] = self.data_dict[key][perm]
            
        
        
        




@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}


def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.
    Args:
      model (`~transformers.PreTrainedModel`): The model to be modified.
      tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
      format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
      resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.
    Returns:
      model (`~transformers.PreTrainedModel`): The modified model.
      tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [chat_format.bos_token, chat_format.eos_token]})
    
    tokenizer.chat_template = chat_format.chat_template

    
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )
    
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())
    
def iter_params(module, recurse=False):
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))
               
def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"], accelerator: "Accelerator", is_peft_model: bool = False
) :
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield model
            add_hooks(model)
    else:
        yield unwrapped_model


def padding_any(data_list, padding_strategy="longest", max_length=1024, padding_side="right", pad_token=0):
    '''
    A padding function to pad arbitrary list or list of tensor. implemented following tokenizer.pad
    '''
    
    if padding_strategy == "longest":
        data_in_max_length = [len(data) for data in data_list if len(data) <= max_length]
        if len(data_in_max_length) == 0:
            return None
        max_len = max(data_in_max_length)
    elif padding_strategy == "max_length":
        max_len = max_length
        
    
    if isinstance(data_list[0], torch.Tensor):
        dtype = data_list[0].dtype
        data_list = [data.detach().cpu().tolist() for data in data_list]
    
    padded_data = []
    for data in data_list:
        if len(data) > max_length:
            continue
        if padding_side == "right":
            padded_data.append(data + [pad_token] * (max_len - len(data)))
        elif padding_side == "left":
            padded_data.append([pad_token] * (max_len - len(data)) + data)
            
    return torch.tensor(padded_data, dtype=dtype)

def pad(tensors, padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


if __name__ == "__main__":
    data_file_path = "YOUR TEST DATA PATH"
    dataset = CustomDataset(data_file_path)
    
    
    

    batch_size = 32  

    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    accelerator = Accelerator(gradient_accumulation_steps=1)
    
    dataloader = accelerator.prepare(dataloader)
    
    
    for batch in dataloader:
        print(batch)
        break
    
    data = {"input_ids": torch.randn(32, 128), "labels": torch.randint(0, 2, (32, 128))}
    
    dataloader = DataLoader(ExpMinDataset(data), batch_size=batch_size, shuffle=True)
    for batch in dataloader:
        print(batch)
        break