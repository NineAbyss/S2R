"""Arguments for training NPC chat planning model."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union
from pathlib import Path
import os 

from transformers import TrainingArguments


@dataclass
class CustomArguments(TrainingArguments):
    """Arguments for the training pipeline."""

    # model
    model_path: Optional[str] = field(
        default=None, metadata={"help": "the path of model."}
    )
    origin_model_path: Optional[str] = field(
        default=None, metadata={"help": "the path of origin model."}
    )

    # data
    data_path: Optional[str] = field(
        default=None, metadata={"help": "the path to folder containing data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "the path of eval data."}
    )
    train_data_path: Optional[str] = field(
        default=None, metadata={"help": "the path of train data."}
    )
    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "the path of test data."}
    )
    sft_data_path: Optional[str] = field(
        default=None, metadata={"help": "the path of sft data."}
    )
    rl_data_path: Optional[str] = field(
        default=None, metadata={"help": "the path of rl data."}
    )
    # tokenizer params
    padding_side: Optional[str] = field(
        default="right",
        metadata={"help": "direction for tokenizer to add padding tokens."},
    )
    truncation_side: Optional[str] = field(
        default="right",
        metadata={"help": "the direction for tokenizer to truncate tokens."},
    )
    max_length: Optional[int] = field(
        default=256, metadata={"help": "the max sentence sequence length."}
    )

    # collater_args
    format_mode: Optional[str] = field(
        default="Baichuan_token",
        metadata={"help": "format to process data, see model architecture."},
    )
    project_name: Optional[str] = field(
        default=None, metadata={"help": "the name of project."}
    )
    prompt_mode: Optional[str] = field(
        default="prefix", metadata={"help": "the mode of prompt."}
    )

    # debug
    debug_mode: Optional[bool] = field(default=False, metadata={"help": "debug mode."})

    # training arguments
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the batch size of train."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the batch size of eval."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the gradient accumulation steps."}
    )
    only_predict_answer: Optional[bool] = field(
        default=True, metadata={"help": "Only predict the answer."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove unused columns."}
    )

    # pooling_type
    prompt_pooling_type: Optional[str] = field(
        default="cls", metadata={"help": "pooling type of prompt."}
    )
    pooling_type: Optional[str] = field(
        default="last", metadata={"help": "pooling type of input."}
    )


@dataclass
class OfflineWeightedPolicyTrainingArguments(CustomArguments):
    pad_labels_with_ignore: Optional[bool] = field(default=False, metadata={"help": "Whether use ignore token to pad labels."})
    use_kl_mask: Optional[bool] = field(default=False)
    lm_kl_coeff: Optional[float] = field(default=0.0)
    clip_range: float = field(default=0.2, metadata={"help": "the range to clip the importance reweighting ratio for policy optimization."})
    lm_sft_coeff: float = field(default=0., metadata={"help": "the coefficient for SFT data language modeling loss."})    
    kl_penalty_mode: str = field(default="kl", metadata={"help": "the calculation mode of kl divergence."})
    train_data_path: List[str] = field(default=None, metadata={"help": "train dataset paths"})
    ref_model_path: Optional[str] = field(default=None, metadata={"help": "the path of reference model."})
    

@dataclass
class RLOOConfig(OfflineWeightedPolicyTrainingArguments):
    r"""
    Configuration class for the [`RLOOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        rloo_k (`int`, *optional*, defaults to `2`):
            REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    reward_model_path: str = None
    num_ppo_epochs: int = 1
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    rloo_k: int = 1
    
    run_name: Optional[str] = None
    dataset_num_proc: Optional[int] = None
    num_mini_batches: int = 1
    total_episodes: Optional[int] = None
    local_rollout_forward_batch_size: int = 64
    num_sample_generations: int = 10
    response_length: int = 53
    stop_token: Optional[Literal["eos"]] = None
    stop_token_id: Optional[int] = None
    temperature: float = 0.7
    # missing_eos_penalty: Optional[float] = None
    sft_model_path: str = "EleutherAI/pythia-160m"
    # world_size: Optional[int] = None
    num_total_batches: Optional[int] = None
    micro_batch_size: Optional[int] = None
    local_batch_size: Optional[int] = None
    dump_data_path: str = None
    restart_step: int = 0
    step_num_per_stage: int = 0
    inner_step_num: int = 1
    local_train_data: Optional[str] = None
    # batch_size: Optional[int] = None
    # local_mini_batch_size: Optional[int] = None
    # mini_batch_size: Optional[int] = None
    use_tf32_forward: bool = False
    use_instance_level: bool = False


@dataclass
class OfflineRLConfig(OfflineWeightedPolicyTrainingArguments):
    r"""
    Configuration class for the [`OfflineRLTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    reward_model_path: str = None
    num_ppo_epochs: int = 1
    whiten_rewards: bool = False
    run_name: Optional[str] = None
    dataset_num_proc: Optional[int] = None
    num_mini_batches: int = 1
    total_episodes: Optional[int] = None
    stop_token: Optional[Literal["eos"]] = None
    stop_token_id: Optional[int] = None
    temperature: float = 0.7
    sft_model_path: str = "EleutherAI/pythia-160m"
    num_total_batches: Optional[int] = None
    micro_batch_size: Optional[int] = None
    local_batch_size: Optional[int] = None
    dump_data_path: str = None
    restart_step: int = 0
    step_num_per_stage: int = 0

    reward_save_path: Union[str, Path] = None
    reward_load_path: Union[str, Path] = None

    use_bonus: bool = False
    use_veri_bonus: bool = False
    debug_mode: bool = False

    use_sft_loss: bool = False
    use_baseline: bool = False
    use_weight: bool = False

    reward_delay_factor: float = 1.0

    use_prefix_baseline: bool = False
    use_level_baseline: bool = False
    use_position_baseline: bool = False
    use_accuracy_baseline: bool = False
    use_softmax_norm: bool = False

    use_process_rl: bool = True
    
    min_group_baseline: int = 10