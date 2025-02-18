from typing import List, Optional

from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    debug_mode: Optional[bool] = field(default=False)

    # data arguments
    data_dir: str = field(default=None, metadata={"help": "the directory to load data."})
    data_paths: List[str] = field(default=None, metadata={"help": "train dataset paths"})
    eval_data_dir: str = field(default=None, metadata={"help": "the directory to load evaluation datasets."})
    eval_data_paths: List[str] = field(default=None, metadata={"help": "evaluation dataset paths."})
    eval_dataset_merge_mode: Optional[str] = field(default='separate',
                                                 metadata={"help": "How to evaluate multiple evalution datasets. Must be one of ['separate', 'merge', 'both']"})

    # model arguments
    model_type: Optional[str] = field(default='qwen', metadata={"help": "base model to use."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained model path"})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "the max sentence sequence length."})

    # tokenizer arguments
    truncation_side: Optional[str] = field(default='left', metadata={"help": "which side to truncate when sequence is too long."})
    padding_side: Optional[str] = field(default='right', metadata={"help": "which side to padding."})
    add_special_tokens: Optional[bool] = field(default=False, metadata={"help": "add bos token and eos token in the text"})

    # save arguments
    save_training_states: Optional[bool] = field(default=False, metadata={"help": "whether or not save training states at the end of the training."})


@dataclass
class SFTTrainingArguments(CustomTrainingArguments):
    data_prompt_name: Optional[str] = field(default='prompt', metadata={"help": "prompt name in data field"})
    data_answer_name: Optional[str] = field(default='answer', metadata={"help": "answer name in data field"})

    # training arguments
    only_predict_answer: Optional[bool] = field(default=True, metadata={"help": "only calculate the loss of answer"})
    pad_labels_with_ignore: Optional[bool] = field(default=False, metadata={"help": "Whether use ignore token to pad labels."})

    
@dataclass
class SFTWeightedTrainingArugments(SFTTrainingArguments):
    data_weight_name: Optional[str] = field(default='weight', metadata={"help": "weight name in data field"})


@dataclass
class SFTWeightedWithKLTrainingArguments(SFTWeightedTrainingArugments):
    use_kl_mask: Optional[bool] = field(default=False)
    lm_kl_coeff: Optional[float] = field(default=0.0)

@dataclass
class OfflineWeightedPolicyTrainingArguments(SFTWeightedWithKLTrainingArguments):
    data_reward_name: str = field(default='reward', metadata={"help": "reward name in data field"})
    clip_range: float = field(default=0.2, metadata={"help": "the range to clip the importance reweighting ratio for policy optimization."})
    lm_sft_coeff: float = field(default=0., metadata={"help": "the coefficient for SFT data language modeling loss."})            