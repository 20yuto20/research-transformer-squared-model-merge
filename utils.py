import re
from copy import deepcopy
from typing import Dict, Optional

import fishfarm
import torch
import torch.utils
import vllm


def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """Load weights from HF transformer model to vLLM model."""

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    num_layers = model.config.num_hidden_layers

    # Load embeddings layer weights.
    model_param = model.get_parameter("model.embed_tokens.weight")
    model_param.copy_(
        param["model.embed_tokens.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )
    model_param = model.get_parameter("lm_head.weight")
    model_param.copy_(
        param["lm_head.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )

    # Load the final layernorm weights.
    model_param = model.get_parameter("model.norm.weight")
    model_param.copy_(
        param["model.norm.weight"].to(model_param.dtype).to(model_param.device)
    )

    for i in range(num_layers):
        # Load qkv_proj weights.
        model_param = model.get_parameter(f"model.layers.{i}.self_attn.qkv_proj.weight")
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.self_attn.q_proj.weight"],
                    param[f"model.layers.{i}.self_attn.k_proj.weight"],
                    param[f"model.layers.{i}.self_attn.v_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load gate_up_proj weights.
        model_param = model.get_parameter(f"model.layers.{i}.mlp.gate_up_proj.weight")
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.mlp.gate_proj.weight"],
                    param[f"model.layers.{i}.mlp.up_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load o_proj and down_proj weights.
        model_param = model.get_parameter(f"model.layers.{i}.self_attn.o_proj.weight")
        model_param.copy_(
            param[f"model.layers.{i}.self_attn.o_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(f"model.layers.{i}.mlp.down_proj.weight")
        model_param.copy_(
            param[f"model.layers.{i}.mlp.down_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load layer_norm weights.
        model_param = model.get_parameter(f"model.layers.{i}.input_layernorm.weight")
        model_param.copy_(
            param[f"model.layers.{i}.input_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.post_attention_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.post_attention_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )


def eval_model(vllm_model, evaluator, ix=None):
    result = evaluator.evaluate(vllm_model, sample_ids=ix)
    return result


def compose_new_params(
    policy,
    param_name,
    decomposed_params,
    learnable_params,
):
    """Compose new parameters from decomposed parameters."""
    mm = policy.get_mask(learnable_params[param_name])
    return (
        decomposed_params[f"{param_name}.U"]
        @ torch.diag_embed(decomposed_params[f"{param_name}.S"] * mm)
        @ decomposed_params[f"{param_name}.V"].T
    ) * (
        decomposed_params[f"{param_name}.S"].sum()
        / (decomposed_params[f"{param_name}.S"] * mm).sum()
    )


@torch.no_grad()
def forward(policy, model, base_params, decomposed_params, learnable_params):
    """Forward pass."""
    new_params = {}
    for k in base_params:
        if "mlp" in k:
            new_params[k] = compose_new_params(
                policy, k, decomposed_params, learnable_params
            )
            model.get_parameter(k).copy_(new_params[k])
        else:
            new_params[k] = base_params[k]
    return new_params


@torch.no_grad()
def load_base_params(
    model,
    base_params,
):
    for k in base_params:
        if "mlp" in k:
            model.get_parameter(k).copy_(base_params[k].cuda())


def backward(
    policy,
    model,
    base_params,
    decomposed_params,
    learnable_params,
):
    """Backward pass."""
    keys_to_backprop = [k for k in base_params if "mlp" in k]
    last_key = keys_to_backprop[-1]
    for k in keys_to_backprop[:-1]:
        compose_new_params(policy, k, decomposed_params, learnable_params).backward(
            model.get_parameter(k).grad, retain_graph=True
        )
    # release graph
    compose_new_params(policy, last_key, decomposed_params, learnable_params).backward(
        model.get_parameter(last_key).grad, retain_graph=False
    )


def classify_samples(vllm_model, test_eval):
    """Classify samples."""

    CLASSIFICATION_PROMPT = """
    # Analyze the given question and classify it into one of four categories: 'code', 'math', 'reasoning' or 'other'. Follow these guidelines:

    1. Code: Questions asking for programming solutions, functions, algorithms. Often includes specific programming terms, language syntax, or data structures.
    2. Math: Questions involving mathematical calculations, formulas, statistics. Often includes numbers, equations, or mathematical operations.
    3. Reasoning: Questions requiring logical thinking, application of scientific knowledge, or critical analysis of information. Often presents statements that need evaluation based on general understanding. 
    4. Other: Questions not clearly fit into above categories.

    Instructions:
    - Consider the primary focus, skills, and knowledge required to answer the question.
    - If a question spans multiple categories, choose the most dominant one.
    - Provide your final classification within \\boxed{} notation. Example: \\boxed{reasoning}

    Format your response as follows:
    Classification: \\boxed{category}
    """

    def extract_classification(text: str) -> Optional[str]:
        """
        Extract the classification from the model's output using regex.
        """
        match = re.search(r"\\boxed{([^}]*)}", text)
        return match.group(1) if match else None

    # Identify the key in the samples that contains the problem text
    problem_key = None
    for key in ("problem", "question", "instruction"):
        if (
            hasattr(test_eval.samples[0], key)
            and getattr(test_eval.samples[0], key) is not None
        ):
            problem_key = key
            break
    assert problem_key is not None, "Could not find problem text in the samples"

    # Prepare classification requests
    classification_requests = [
        fishfarm.models.GenerationRequest(
            messages=[
                fishfarm.Message("system", CLASSIFICATION_PROMPT),
                fishfarm.Message("user", getattr(sample, problem_key)),
            ]
        )
        for sample in test_eval.samples
    ]

    # Generate classifications using the model
    model_outputs = vllm_model.generate(classification_requests)

    # Process results and update samples
    classified_samples = []
    for sample, result in zip(test_eval.samples, model_outputs):
        prediction = extract_classification(result.generation)
        if prediction not in ["code", "math", "reasoning"]:
            prediction = "other"
        sample.expert_label = prediction
        classified_samples.append(sample)

    return classified_samples


def eval_model_experts_prompt_based(
    vllm_model,
    evaluator,
    experts_path_dict,
    policy,
    model,
    base_params,
    decomposed_params,
    task_metric,
):
    """Evaluate the model using expert models and prompt-based classification."""
    results_by_expert: Dict[str, Dict] = {}

    # Classify all test samples
    classified_samples = classify_samples(vllm_model, evaluator)

    # Evaluate samples for each expert model
    for expert_label, expert_model_path in experts_path_dict.items():
        # Filter samples for current expert
        expert_samples = [
            sample
            for sample in classified_samples
            if sample.expert_label == expert_label
        ]
        if not expert_samples:
            continue

        # Update test evaluation with filtered samples
        evaluator.samples = expert_samples

        # Load and apply expert model parameters if available
        if expert_model_path:
            policy.load_state_dict(torch.load(expert_model_path))
            expert_params = policy.get_learnable_params()
            updated_params = forward(
                policy=policy,
                model=model,
                base_params=base_params,
                decomposed_params=decomposed_params,
                learnable_params=expert_params,
            )
            load_hf_params_to_vllm(updated_params, vllm_model.llm)

        # Evaluate current expert model
        evaluation_results = eval_model(vllm_model, evaluator)

        # Store results for current expert
        results_by_expert[expert_label] = {
            "num_samples": len(expert_samples),
            "test_acc": evaluation_results.aggregate_metrics[task_metric],
        }

    # Compute the overall accuracy.
    data_dict = deepcopy(results_by_expert)
    data_dict["final_test_acc"] = 0.0
    for label in results_by_expert.keys():
        data_dict["final_test_acc"] += (
            results_by_expert[label]["test_acc"]
            * results_by_expert[label]["num_samples"]
        )
    data_dict["final_test_acc"] /= len(classified_samples)

    return data_dict

def eval_model_experts_prompt_based_enhanced(  
    vllm_model,  
    evaluator,  
    experts_path_dict,  
    policy,  
    model,  
    base_params,  
    decomposed_params,  
    task_metric,  
    use_enhanced_merging=True,  
):  
    """拡張された専門知識ベクトルマージを使用したモデル評価"""  
    results_by_expert = {}  
    
    # サンプルの分類  
    classified_samples = classify_samples(vllm_model, evaluator)  
    
    # 分類結果の分析と重み付け係数の計算  
    expert_weights = {}  
    for expert_label in experts_path_dict.keys():  
        # この専門家に分類されたサンプル数  
        expert_samples = [s for s in classified_samples if s.expert_label == expert_label]  
        if not expert_samples:  
            expert_weights[expert_label] = 0.0  
            continue  
        
        # サンプル数に基づく重み付け（他の方法も可能）  
        expert_weights[expert_label] = len(expert_samples) / len(classified_samples)  
    
    # 専門知識ベクトルを読み込む  
    experts_params = {}  
    for expert_label, expert_path in experts_path_dict.items():  
        if not expert_path or expert_weights[expert_label] <= 0:  
            continue  
        
        # 専門知識ベクトルのロード  
        policy.load_state_dict(torch.load(expert_path))  
        experts_params[expert_label] = policy.get_learnable_params()  
    
    # 拡張されたマージ関数を使用  
    if use_enhanced_merging and len(experts_params) > 1:  
        merged_params = merge_expert_vectors(  
            experts_params,   
            expert_weights,  
            decomposed_params,  
            similarity_threshold=0.8,  
            adaptive_scaling=True  
        )  
        
        # マージされたパラメータを使用してモデルを更新  
        updated_params = {}  
        for k in base_params:  
            if "mlp" in k and k in merged_params:  
                updated_params[k] = compose_new_params_enhanced(  
                    policy=policy,  
                    param_name=k,  
                    decomposed_params=decomposed_params,  
                    learnable_params=merged_params,  
                )  
            else:  
                updated_params[k] = base_params[k]  
        
        # VLLMモデルに更新されたパラメータをロード  
        load_hf_params_to_vllm(updated_params, vllm_model.llm)  
        
        # モデルを評価  
        evaluation_results = eval_model(vllm_model, evaluator)  
        results_by_expert["merged"] = {  
            "num_samples": len(classified_samples),  
            "test_acc": evaluation_results.aggregate_metrics[task_metric],  
        }  
    
    # 結果の計算と返却  
    data_dict = results_by_expert.copy()  
    data_dict["expert_weights"] = expert_weights  
    data_dict["final_test_acc"] = results_by_expert.get("merged", {"test_acc": 0})["test_acc"]  
    
    return data_dict

def compose_new_params_enhanced(  
    policy,  
    param_name,  
    decomposed_params,  
    learnable_params,  
    task_specific_scaling=None,  
):  
    """拡張されたパラメータ合成関数"""  
    mm = policy.get_mask(learnable_params[param_name])  

    # タスク特有のスケーリングがある場合は適用  
    if task_specific_scaling is not None:  
        mm = mm * task_specific_scaling.get(param_name, 1.0)  

    # SVD成分の合成と正規化  
    composed_param = (  
        decomposed_params[f"{param_name}.U"]  
        @ torch.diag_embed(decomposed_params[f"{param_name}.S"] * mm)  
        @ decomposed_params[f"{param_name}.V"].T  
    )  
    
    # 正規化係数の計算  
    original_sum = decomposed_params[f"{param_name}.S"].sum()  
    modified_sum = (decomposed_params[f"{param_name}.S"] * mm).sum()  
    
    # より安定した正規化（ゼロ除算を防ぐ）  
    scaling_factor = original_sum / (modified_sum + 1e-6)  
    
    return composed_param * scaling_factor

@torch.no_grad()  
def merge_expert_vectors(  
    experts_dict,  
    weights_dict,  
    decomposed_params,  
    similarity_threshold=0.7,  
    adaptive_scaling=True,  
):  
    """複数の専門知識ベクトルをマージして最適化する関数"""  
    merged_params = {}  
    
    # パラメータキーに対してループ  
    for param_key in list(experts_dict.values())[0].keys():  
        # 各エキスパートからのマスク値を収集  
        expert_masks = []  
        for expert_name, expert_params in experts_dict.items():  
            weight = weights_dict.get(expert_name, 1.0)  
            expert_masks.append(weight * expert_params[param_key])  
        
        # 類似度に基づくクラスタリングとマージ  
        if similarity_threshold < 1.0:  
            # ここに類似度ベースのクラスタリングを実装  
            # 類似した専門知識ベクトルをグループ化  
            pass  
        
        # マスク値の結合（単純な加重平均ではなく、パラメータごとに最適なマスク値を選択）  
        if adaptive_scaling:  
            # より高度なマージロジック（例：最大値を取る、特定パターンを優先する等）  
            merged_mask = torch.stack(expert_masks).max(dim=0).values  
        else:  
            # 従来の加重平均方式  
            merged_mask = torch.zeros_like(expert_masks[0])  
            for mask in expert_masks:  
                merged_mask += mask  
            merged_mask /= len(expert_masks)  
        
        # マージされたマスクを使って新しいパラメータを合成  
        merged_params[param_key] = merged_mask  
    
    return merged_params