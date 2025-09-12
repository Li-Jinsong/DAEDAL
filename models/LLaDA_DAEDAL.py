import logging
import os
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple, Union, TypeVar
import torch
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from packaging import version
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from dllm_eval.api.instance import Instance
from dllm_eval.api.model import LM, TemplateLM
from dllm_eval.api.registry import register_model
from dllm_eval.models.utils import get_dtype, configure_pad_token
from dllm_eval.models.modeling_llada import LLaDAModelLM


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    initial_gen_length=64,
    max_gen_length=2048,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    high_conf_threshold=0.90,
    low_conf_threshold=0.10,
    expansion_factor=8,
    mask_id=126336,
    eos_token_id=126081,
    eos_confidence_threshold=0.5,
    expand_eos_confidence_threshold=0.9,
    eos_check_tokens=32,
):
    def _calculate_eos_confidence(logits, total_lengths, prompt_length, eos_check_tokens):
        if eos_token_id is None:
            return torch.zeros(logits.shape[0], device=logits.device)
        confidences = F.softmax(logits, dim=-1)
        predicted_tokens = torch.argmax(logits, dim=-1)
        batch_eos_confidences = []
        for i in range(logits.shape[0]):
            eos_confs_for_avg = []
            start_scan_pos = total_lengths[i].item() - 1
            end_scan_pos = prompt_length - 1
            for pos in range(start_scan_pos, end_scan_pos, -1):
                if len(eos_confs_for_avg) >= eos_check_tokens:
                    break
                if predicted_tokens[i, pos] == eos_token_id:
                    eos_confs_for_avg.append(confidences[i, pos, eos_token_id].item())
            avg_conf = sum(eos_confs_for_avg) / eos_check_tokens
            batch_eos_confidences.append(avg_conf)
        return torch.tensor(batch_eos_confidences, device=logits.device)

    with torch.autocast(device_type="cuda"):
        batch_size = prompt.shape[0]
        device = prompt.device
        prompt_length = prompt.shape[1]
        assert eos_token_id is not None
        gen_lengths = torch.full((batch_size,), initial_gen_length, dtype=torch.long, device=device)
        x = torch.full(
            (batch_size, prompt_length + initial_gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, :prompt_length] = prompt.clone()
        prompt_index = x != mask_id
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print("[Stage-1] Initial Length Adjustment")
        while True:
            total_lengths = prompt_length + gen_lengths
            max_len_pre = x.shape[1]
            arange_tensor_pre = torch.arange(max_len_pre, device=device).expand(batch_size, -1)
            attention_mask_pre = (arange_tensor_pre < total_lengths.unsqueeze(1)).long()
            logits_pre = model(x, attention_mask=attention_mask_pre).logits
            batch_eos_confidences = _calculate_eos_confidence(logits_pre, total_lengths, prompt_length, eos_check_tokens)
            sequences_to_expand = (batch_eos_confidences < eos_confidence_threshold) & (gen_lengths < max_gen_length)
            if not sequences_to_expand.any():
                if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(f"All sequences' EOS confidence reach the threshold {eos_confidence_threshold} or max length.")
                break
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                 print(f"Some sequences' EOS confidence ({[round(c.item(), 4) for c in batch_eos_confidences]}) < {eos_confidence_threshold}. Expand initial length.")
            max_new_gen_len = gen_lengths[sequences_to_expand].max().item()
            new_gen_lengths = gen_lengths.clone()
            new_gen_lengths[sequences_to_expand] = torch.clamp(gen_lengths[sequences_to_expand] + expansion_factor, max=max_gen_length)
            if new_gen_lengths.max() <= gen_lengths.max():
                 if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(f"WARNING: Cannot expand initial length further (already at max length: {max_gen_length}).")
                 break
            max_new_total_len = prompt_length + new_gen_lengths.max()
            new_x_tensor = torch.full((batch_size, max_new_total_len), eos_token_id, dtype=torch.long, device=device)
            for i in range(batch_size):
                original_total_len = prompt_length + gen_lengths[i].item()
                new_x_tensor[i, :original_total_len] = x[i, :original_total_len]
                if sequences_to_expand[i]:
                    new_total_len_i = prompt_length + new_gen_lengths[i].item()
                    new_x_tensor[i, original_total_len : new_total_len_i] = mask_id
            x = new_x_tensor
            gen_lengths = new_gen_lengths
        new_gen_lengths_with_eos = gen_lengths + int(eos_check_tokens/2)
        new_gen_lengths_with_eos = torch.clamp(new_gen_lengths_with_eos, max=max_gen_length)
        max_new_total_len = prompt_length + new_gen_lengths_with_eos.max()
        intermediate_x_tensor = torch.full((batch_size, max_new_total_len), eos_token_id, dtype=torch.long, device=device)
        for i in range(batch_size):
            intermediate_x_tensor[i, :prompt_length] = x[i, :prompt_length]
            intermediate_x_tensor[i, prompt_length:prompt_length+gen_lengths[i].item()] = mask_id
        x = intermediate_x_tensor
        gen_lengths = new_gen_lengths_with_eos
        
        
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[Stage-2] Iterative Denoising and Mask Insertion")

        current_pos = torch.full((batch_size,), prompt_length, dtype=torch.long, device=device)
        denoise_only_mode = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        while (current_pos < prompt_length + gen_lengths).any():
            total_lengths = prompt_length + gen_lengths
            x_before_step = x.clone()
            
            for i in range(batch_size):
                if gen_lengths[i] >= max_gen_length and not denoise_only_mode[i]:
                    if current_pos[i] < total_lengths[i]:
                        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                            print(f"Sequence {i} has reached the max length {max_gen_length}. Entering denoise-only mode.")
                        denoise_only_mode[i] = True

            max_len = x.shape[1]
            arange_tensor = torch.arange(max_len, device=device).expand(batch_size, -1)
            attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()
            if cfg_scale > 0.0:
                un_x = x.clone(); un_x[prompt_index] = mask_id
                un_attention_mask = attention_mask.clone()
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = torch.cat([attention_mask, un_attention_mask], dim=0)
                logits, un_logits = torch.chunk(model(x_, attention_mask=attention_mask_).logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            predicted_tokens = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            confidences = F.softmax(logits, dim=-1)
            predicted_confidences = torch.gather(confidences, dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(-1)
            batch_eos_confidences = _calculate_eos_confidence(logits, total_lengths, prompt_length, eos_check_tokens)

            block_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i in range(batch_size):
                if current_pos[i] >= total_lengths[i]: continue
                block_mask[i, current_pos[i]:min(current_pos[i] + block_length, total_lengths[i].item())] = True
            
            currently_masked = (x == mask_id)
            high_conf_indices = (predicted_confidences > high_conf_threshold) & block_mask & currently_masked & (predicted_tokens != mask_id)

            for i in range(batch_size):
                if current_pos[i] >= total_lengths[i]: continue
                start_idx, end_idx = current_pos[i], min(current_pos[i] + block_length, total_lengths[i].item())
                if not high_conf_indices[i, start_idx:end_idx].any():
                    valid_fallback_mask = block_mask[i] & currently_masked[i]
                    if not valid_fallback_mask.any(): continue
                    candidate_indices = torch.where(valid_fallback_mask)[0]
                    candidate_confs = predicted_confidences[i, candidate_indices]
                    candidate_tokens = predicted_tokens[i, candidate_indices]
                    sorted_confs, sort_indices = torch.sort(candidate_confs, descending=True)
                    best_idx_to_fill = -1
                    for sorted_idx in sort_indices:
                        if candidate_tokens[sorted_idx] != mask_id:
                            best_idx_to_fill = candidate_indices[sorted_idx]; break
                    if best_idx_to_fill != -1:
                        high_conf_indices[i, best_idx_to_fill] = True
                    else:
                        stuck_logits = logits[i, candidate_indices]
                        stuck_logits[:, mask_id] = -torch.inf
                        new_confidences = F.softmax(stuck_logits, dim=-1)
                        new_best_confs, new_best_tokens = torch.max(new_confidences, dim=-1)
                        best_of_the_best_local_idx = torch.argmax(new_best_confs)
                        pos_to_fill = candidate_indices[best_of_the_best_local_idx]
                        token_to_fill = new_best_tokens[best_of_the_best_local_idx]
                        predicted_tokens[i, pos_to_fill] = token_to_fill
                        high_conf_indices[i, pos_to_fill] = True

            potential_expand_mask = (predicted_confidences < low_conf_threshold) & block_mask & currently_masked & (~high_conf_indices)
            expand_indices = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i in range(batch_size):
                if batch_eos_confidences[i] >= expand_eos_confidence_threshold or gen_lengths[i] >= max_gen_length: continue
                if denoise_only_mode[i] or current_pos[i] >= total_lengths[i]: continue
                masked_candidates = torch.where(potential_expand_mask[i])[0]
                if len(masked_candidates) > 0:
                    candidate_confs = predicted_confidences[i, masked_candidates]
                    num_to_expand = min(1, len(masked_candidates))
                    if num_to_expand > 0:
                        _, lowest_conf_local_indices = torch.topk(candidate_confs, num_to_expand, largest=False)
                        indices_to_expand_global = masked_candidates[lowest_conf_local_indices]
                        expand_indices[i, indices_to_expand_global] = True
            
            fill_mask = high_conf_indices
            if not expand_indices.any():
                x[fill_mask] = predicted_tokens[fill_mask]
            else:
                x[fill_mask] = predicted_tokens[fill_mask]
                max_new_total_len = 0
                temp_new_gen_lengths = gen_lengths.clone()
                for i in range(batch_size):
                    expansion_count = expand_indices[i].sum().item()
                    if expansion_count > 0:
                        new_len = gen_lengths[i].item() + expansion_count * (expansion_factor - 1)
                        temp_new_gen_lengths[i] = min(new_len, max_gen_length)
                max_new_total_len = prompt_length + temp_new_gen_lengths.max()
                
                new_x_tensor = torch.full((batch_size, max_new_total_len), eos_token_id, device=device, dtype=torch.long)
                new_gen_lengths = torch.zeros_like(gen_lengths)

                for i in range(batch_size):
                    if not expand_indices[i].any():
                        total_len = prompt_length + gen_lengths[i].item()
                        new_x_tensor[i, :total_len] = x[i, :total_len]
                        new_gen_lengths[i] = gen_lengths[i]
                        continue
                    write_ptr = prompt_length
                    new_x_tensor[i, :prompt_length] = x[i, :prompt_length]
                    for j in range(prompt_length, prompt_length + gen_lengths[i].item()):
                        if write_ptr >= max_new_total_len: break
                        if expand_indices[i, j]:
                            end_write = min(write_ptr + expansion_factor, max_new_total_len)
                            new_x_tensor[i, write_ptr:end_write] = mask_id
                            write_ptr = end_write
                        else:
                            new_x_tensor[i, write_ptr] = x[i, j]
                            write_ptr += 1
                    new_gen_lengths[i] = write_ptr - prompt_length
                x = new_x_tensor
                gen_lengths = new_gen_lengths
            for i in range(batch_size):
                total_len = prompt_length + gen_lengths[i]
                while current_pos[i] < total_len:
                    start_check = current_pos[i]
                    end_check = min(start_check + block_length, total_len.item())
                    if start_check == end_check: break
                    if not (x[i, start_check:end_check] == mask_id).any():
                        current_pos[i] = start_check + block_length
                    else:
                        break
            if torch.equal(x, x_before_step):
                if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(f"WARNING: Sequence state is stagnant, forcing generation to end.")
                break
        final_outputs = []
        for i in range(batch_size):
            final_len = prompt_length + gen_lengths[i]
            final_outputs.append(x[i, :final_len])
        return final_outputs


@register_model("LLaDA_DAEDAL")
class LLaDA_DAEDAL(TemplateLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 20480

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        escape_until:Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        mc_num: int = 1024,
        remasking: str = "expand",
        mask_id: int = 126336,
        is_check_greedy : bool =True,
        assistant_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mc_num = mc_num
        self.mask_id = mask_id
        self.remasking = remasking
        self.pretrained = pretrained
        self.is_check_greedy = is_check_greedy
        self.assistant_prefix = assistant_prefix
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))
            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator
            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()
            if not (parallelize or accelerator.num_processes > 1):
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )
            revision = str(revision)
            revision = revision + ("/" + subfolder if subfolder is not None else "")
            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                **kwargs,
            )
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )
        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > self.accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {self.accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )
                    self._device = torch.device(f"{self.accelerator.device}")
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
            else:
                self._rank = 0
                self._world_size = 1
        else:
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1
             
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        self.is_first_inference = True

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if parallelize is None and gpus is not None and gpus > 1:
            parallelize = True
        args = {}
        if parallelize:
            max_memory_all_gpus = get_max_memory()
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_memory_per_gpu_map = {
                device_idx: max_memory_per_gpu for device_idx in range(len(max_memory_all_gpus))
            } if max_memory_per_gpu is not None else {k: v for k, v in max_memory_all_gpus.items()}
            if hasattr(self, "accelerator"):
                max_memory_per_gpu_map = {
                    k: v for k, v in max_memory_all_gpus.items() if k % num_local_processes == self.accelerator.process_index % num_local_processes
                }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto"
            args["offload_folder"] = offload_folder
            if max_cpu_memory is not None:
                args["max_memory"]["cpu"] = max_cpu_memory
            eval_logger.info(
                f"Model parallel set to True. Max memory per GPU: {args['max_memory']}, Device map: {args['device_map']}"
            )
        else:
            args["device_map"] = {"": str(self.device)}
            eval_logger.info(
                f"Model parallel set to False. Device map: {args['device_map']}"
            )
        return args

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length > 1e10:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]
        if backend != "default":
            self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "seq2seq"
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
            else:
                eval_logger.warning(
                    "HF model type is neither CausalLM nor Seq2SeqLM. Assuming CausalLM."
                )
                self.backend = "causal"
        if self.AUTO_MODEL_CLASS is None:
            if self.backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif self.backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        if autogptq or gptqmodel or peft or delta:
            raise NotImplementedError("Advanced model loading options are not implemented for this custom class.")
        model_kwargs = kwargs if kwargs else {}
        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
            )
        )
        self._model = LLaDAModelLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        if not parallelize:
             self._model = self._model.to(self.device)
        self._model.eval()

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast_tokenizer
        }
        if add_bos_token:
            kwargs["add_bos_token"] = True
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, **kwargs)
            else:
                self.tokenizer = tokenizer
        else:
            model_name = pretrained if isinstance(pretrained, str) else self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs["add_special_tokens"] = self.add_bos_token
        else:
            special_tokens_kwargs["add_special_tokens"] = add_special_tokens
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        add_special_tokens = {"add_special_tokens": self.add_bos_token} if self.backend == "causal" else {}
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len and encoding["input_ids"].size(1) > left_truncate_len:
            eval_logger.warning(
                f"Left-truncating from {encoding['input_ids'].size(1)} to {left_truncate_len} tokens."
            )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if self.backend == "seq2seq":
                return self.model(input_ids=inps, attention_mask=attn_mask, labels=labels).logits
            else:
                return self.model(inps, attention_mask=attn_mask).logits

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError
         
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running generate_until requests")
        ds_data = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds_data)
        gen_kwargs = requests[0].args[1]
        for batch in ds.iter(batch_size=int(self.batch_size)):
            contexts = batch["text"]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
            )
            prompt_length = context_enc.shape[1]
            out_list = generate(
                model=self.model,
                prompt=context_enc,
                tokenizer=self.tokenizer,
                initial_gen_length=gen_kwargs.get("initial_gen_length", 64),
                max_gen_length=gen_kwargs.get("max_gen_length", 2048),
                block_length=gen_kwargs.get("block_length", 32),
                temperature=gen_kwargs.get("temperature", 0.0),
                cfg_scale=gen_kwargs.get("cfg_scale", 0.0),
                high_conf_threshold=gen_kwargs.get("high_conf_threshold", 0.90),
                low_conf_threshold=gen_kwargs.get("low_conf_threshold", 0.10),
                expansion_factor=gen_kwargs.get("expansion_factor", 8),
                mask_id=self.mask_id,
                eos_token_id=self.eot_token_id,
                eos_confidence_threshold=gen_kwargs.get("eos_confidence_threshold", 0.5),
                expand_eos_confidence_threshold=gen_kwargs.get("expand_eos_confidence_threshold", 0.9),
                eos_check_tokens=gen_kwargs.get("eos_check_tokens", 32),
            )
            cont_toks_list = []
            for single_output in out_list:
                generated_tokens = single_output[prompt_length:]
                decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                cont_toks_list.append(decoded_text)
            if self.rank == 0 and self.is_first_inference:
                eval_logger.info("\n\n--- First Batch Inference (Rank 0) ---")
                for i, (question, answer) in enumerate(zip(contexts, cont_toks_list)):
                    eval_logger.info(f"Question {i+1}: {question}")
                    eval_logger.info(f"\nAnswer   {i+1}: {answer}\n")
                eval_logger.info("------------------------------------\n\n")
                self.is_first_inference = False
            for s in cont_toks_list:
                if not self.escape_until:
                    stop_sequences = gen_kwargs.get("until", [])
                    if stop_sequences:
                        for term in stop_sequences:
                            if len(term) > 0:
                                s = s.split(term)[0]
                res.append(s)
                bar.update(1)
        bar.close()
        return res

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.assistant_prefix:
            chat_templated += self.assistant_prefix
        return chat_templated