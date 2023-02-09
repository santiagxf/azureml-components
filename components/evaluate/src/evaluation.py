import mlflow
import os
from datasets import load_dataset
from jobtools.arguments import StringEnum

class ModelType(StringEnum):
    REGRESSOR="regressor"
    CLASSIFIER="classifier"

class ModelEvaluator(StringEnum):
    DEFAULT="default"

class DataFormat(StringEnum):
    CSV="csv"
    PARQUET="parquet"
    JSON="json"

def evaluate(model: str, evaluation_data_path: str, evaluation_data_format: DataFormat, target: str, 
             model_type: ModelType, evaluator: ModelEvaluator, evaluation_results: str, baseline: str = None):
    eval_data = load_dataset(str(evaluation_data_format), data_files={'eval': os.path.join(evaluation_data_path, "*")})
    result = mlflow.evaluate(
        baseline_model=baseline,
        model=model,
        data=eval_data["eval"].data.to_pandas(),
        targets=target,
        model_type=str(model_type),
        evaluators=[str(evaluator)],
        env_manager="local",
    )

    result.save(evaluation_results)