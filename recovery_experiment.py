"""Layer/module precision-recovery experiments for the MCLM dataset."""

import argparse
import copy
import gc
import json
import os
import random
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM

import main as evaluation
from tools import build_samples, get_dataset, load_model


STAGE_ONE_MODULES = {
    "down_proj_all": ("mlp.down_proj",),
    "up_proj_all": ("mlp.up_proj",),
    "gate_proj_all": ("mlp.gate_proj",),
    "mlp_all": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "o_proj_all": ("self_attn.o_proj",),
    "qkv_proj_all": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "attention_all": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FP16 module-recovery experiments on a quantized MCLM model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp16_checkpoint", required=True)
    parser.add_argument("--quant_type", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--languages", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--enable_thinking",
        type=evaluation.str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--num_layer_groups", type=int, default=4)
    parser.add_argument(
        "--stages",
        default="1,2,3",
        help="Comma-separated recovery stages to run (default: 1,2,3).",
    )
    parser.add_argument(
        "--save_path",
        default="results",
        help="Root result directory used by the recovery runner.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_attribute(root, dotted_path):
    current = root
    for name in dotted_path.split("."):
        if not hasattr(current, name):
            raise AttributeError(
                f"{type(current).__name__} has no attribute {name!r} "
                f"while resolving {dotted_path!r}."
            )
        current = getattr(current, name)
    return current


def get_parent_and_name(root, dotted_path):
    parts = dotted_path.split(".")
    parent = root
    for name in parts[:-1]:
        if not hasattr(parent, name):
            raise AttributeError(
                f"Cannot resolve recovery module {dotted_path!r}: "
                f"missing {name!r}."
            )
        parent = getattr(parent, name)
    return parent, parts[-1]


def find_decoder_layers(model):
    candidate_paths = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "transformer.blocks",
    )
    for path in candidate_paths:
        try:
            layers = get_attribute(model, path)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and len(layers) > 0:
            return layers, path
    raise ValueError(
        "Cannot locate decoder layers. Add this model architecture to "
        "find_decoder_layers()."
    )


def make_equal_layer_groups(num_layers, num_groups):
    if num_groups < 1:
        raise ValueError("--num_layer_groups must be at least 1.")
    if num_groups > num_layers:
        raise ValueError(
            f"Cannot split {num_layers} layers into {num_groups} non-empty groups."
        )

    base_size, remainder = divmod(num_layers, num_groups)
    groups = []
    start = 0
    for group_index in range(num_groups):
        size = base_size + (1 if group_index < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def make_experiments(num_layers, groups, stages):
    all_layers = list(range(num_layers))
    experiments = [{
        "name": "quant_baseline",
        "stage": 0,
        "layers": [],
        "modules": [],
    }]

    if 1 in stages:
        for name, modules in STAGE_ONE_MODULES.items():
            experiments.append({
                "name": name,
                "stage": 1,
                "layers": all_layers,
                "modules": list(modules),
            })

    if 2 in stages:
        for index, group in enumerate(groups):
            experiments.append({
                "name": f"down_group_{index}",
                "stage": 2,
                "layers": group,
                "modules": ["mlp.down_proj"],
            })

    if 3 in stages:
        for count in range(1, len(groups) + 1):
            prefix_layers = [
                layer for group in groups[:count] for layer in group
            ]
            suffix_layers = [
                layer for group in groups[-count:] for layer in group
            ]
            experiments.append({
                "name": f"down_prefix_{count}",
                "stage": 3,
                "direction": "prefix",
                "group_count": count,
                "layers": prefix_layers,
                "modules": ["mlp.down_proj"],
            })
            # The full prefix and full suffix are identical, so run it once.
            if count < len(groups):
                experiments.append({
                    "name": f"down_suffix_{count}",
                    "stage": 3,
                    "direction": "suffix",
                    "group_count": count,
                    "layers": suffix_layers,
                    "modules": ["mlp.down_proj"],
                })

    return experiments


def load_fp16_reference(checkpoint):
    print(f"Loading FP16 reference model on CPU: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def module_device(module):
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def recover_modules(quant_model, reference_model, layer_indices, module_paths):
    quant_layers, quant_layer_path = find_decoder_layers(quant_model)
    reference_layers, reference_layer_path = find_decoder_layers(reference_model)
    if len(quant_layers) != len(reference_layers):
        raise ValueError(
            "Quantized and FP16 models have different layer counts: "
            f"{len(quant_layers)} vs {len(reference_layers)}."
        )

    recovered_parameters = 0
    recovered_modules = 0
    for layer_index in layer_indices:
        quant_layer = quant_layers[layer_index]
        reference_layer = reference_layers[layer_index]

        for module_path in module_paths:
            target_parent, target_name = get_parent_and_name(
                quant_layer, module_path
            )
            if not hasattr(target_parent, target_name):
                raise AttributeError(
                    f"Layer {layer_index} under {quant_layer_path} does not "
                    f"contain {module_path!r}."
                )

            source_module = get_attribute(reference_layer, module_path)
            target_module = getattr(target_parent, target_name)
            target_device = module_device(target_module)

            recovered_module = copy.deepcopy(source_module)
            recovered_module.to(device=target_device, dtype=torch.float16)
            recovered_module.eval()

            recovered_parameters += sum(
                parameter.numel()
                for parameter in source_module.parameters()
            )
            recovered_modules += 1
            setattr(target_parent, target_name, recovered_module)
            del target_module

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "quant_layer_path": quant_layer_path,
        "reference_layer_path": reference_layer_path,
        "recovered_modules_count": recovered_modules,
        "recovered_parameters": recovered_parameters,
    }


def make_model_args(args):
    """Create the argument namespace expected by tools.load_model()."""
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        quant_type=args.quant_type,
        model_type=args.model_type,
    )


def evaluate_experiment(
    args,
    experiment,
    model,
    tokenizer,
    dataset_full,
    dataset_prompt,
    output_dir,
):
    language_results = {}
    total_tokens = 0
    total_instances = 0

    model.generation_config.do_sample = False

    for lang in args.languages.split(","):
        print(f"[{experiment['name']}] Evaluating {lang}")
        samples = build_samples(dataset_full, lang, dataset_prompt)
        records = evaluation.inference(
            samples=samples,
            tokenizer=tokenizer,
            model=model,
            model_type=args.model_type,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            record_output_tokens=True,
        )
        accuracy = evaluation.evaluate(records)
        length_metrics = evaluation.calculate_mclm_length_metrics(records)
        result = {
            "accuracy": accuracy,
            "samples": len(records),
            "generation_length_metrics": length_metrics,
            "records": records,
        }
        language_results[lang] = result
        total_tokens += length_metrics["total_generated_tokens"]
        total_instances += length_metrics["instances"]

        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, f"{lang}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    macro_accuracy = sum(
        result["accuracy"] for result in language_results.values()
    ) / len(language_results)
    return {
        "macro_accuracy": macro_accuracy,
        "total_generated_tokens": total_tokens,
        "mean_generated_tokens": total_tokens / total_instances,
        "generation_instances": total_instances,
        "languages": {
            lang: {
                "accuracy": result["accuracy"],
                "samples": result["samples"],
                "mean_generated_tokens": result[
                    "generation_length_metrics"
                ]["mean_generated_tokens"],
            }
            for lang, result in language_results.items()
        },
    }


def main():
    args = parse_args()
    if args.quant_type == "fp16":
        raise ValueError(
            "Recovery experiments require a quantized --checkpoint; "
            "--quant_type cannot be fp16."
        )

    stages = {
        int(value.strip())
        for value in args.stages.split(",")
        if value.strip()
    }
    if not stages or not stages.issubset({1, 2, 3}):
        raise ValueError("--stages must contain only 1, 2 and/or 3.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    languages = args.languages.split(",")
    dataset_full, dataset_prompt = get_dataset("mclm", languages)
    reference_model = load_fp16_reference(args.fp16_checkpoint)
    reference_layers, layer_path = find_decoder_layers(reference_model)
    num_layers = len(reference_layers)
    groups = make_equal_layer_groups(num_layers, args.num_layer_groups)
    experiments = make_experiments(num_layers, groups, stages)
    total_reference_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )

    recovery_root = os.path.join(
        args.save_path,
        "recovery",
        "mclm",
        args.model_type,
        args.quant_type,
    )
    os.makedirs(recovery_root, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "fp16_checkpoint": args.fp16_checkpoint,
        "quant_type": args.quant_type,
        "model_type": args.model_type,
        "languages": languages,
        "num_layers": num_layers,
        "decoder_layer_path": layer_path,
        "num_layer_groups": args.num_layer_groups,
        "layer_groups": groups,
        "stages": sorted(stages),
        "experiments": {},
    }

    quant_baseline = None
    for experiment_index, experiment in enumerate(experiments, start=1):
        print(
            f"\nExperiment {experiment_index}/{len(experiments)}: "
            f"{experiment['name']}"
        )
        tokenizer, quant_model = load_model(make_model_args(args))

        recovery_metadata = {
            "recovered_modules_count": 0,
            "recovered_parameters": 0,
        }
        if experiment["modules"]:
            recovery_metadata = recover_modules(
                quant_model,
                reference_model,
                experiment["layers"],
                experiment["modules"],
            )

        recovered_parameters = recovery_metadata["recovered_parameters"]
        recovered_parameter_ratio = (
            recovered_parameters / total_reference_parameters
        )
        experiment_dir = os.path.join(recovery_root, experiment["name"])
        aggregate = evaluate_experiment(
            args,
            experiment,
            quant_model,
            tokenizer,
            dataset_full,
            dataset_prompt,
            experiment_dir,
        )

        result_summary = {
            **experiment,
            **recovery_metadata,
            "recovered_parameter_ratio": recovered_parameter_ratio,
            **aggregate,
        }
        if experiment["name"] == "quant_baseline":
            quant_baseline = aggregate
            result_summary["accuracy_difference"] = 0.0
            result_summary["mean_generated_tokens_difference"] = 0.0
        else:
            result_summary["accuracy_difference"] = (
                aggregate["macro_accuracy"]
                - quant_baseline["macro_accuracy"]
            )
            result_summary["mean_generated_tokens_difference"] = (
                aggregate["mean_generated_tokens"]
                - quant_baseline["mean_generated_tokens"]
            )

        summary["experiments"][experiment["name"]] = result_summary
        with open(
            os.path.join(experiment_dir, "metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(recovery_root, "recovery_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        del quant_model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del reference_model
    gc.collect()
    print(
        "\nRecovery experiments completed. Summary: "
        f"{os.path.join(recovery_root, 'recovery_summary.json')}"
    )


if __name__ == "__main__":
    main()
