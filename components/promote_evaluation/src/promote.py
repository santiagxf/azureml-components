import mlflow
import tempfile
import os
import time
import logging

from mlflow.entities import Param, Metric
from mlflow.models import EvaluationResult

def promote_to_parent(evaluation_results: str, 
                      include_all_tracking: bool = False, 
                      include_all_artifacts: bool = False
                     ):
    logger = logging.getLogger()

    with mlflow.start_run(nested=True) as active_run:
        active_run_id = active_run.info.run_id
        parent_run_id = active_run.data.tags["mlflow.parentRunId"]
        experiment_id = active_run.info.experiment_id
        logger.debug(f"Active run id: {active_run_id}")
        logger.debug(f"Parent run id: {parent_run_id}")

        client = mlflow.MlflowClient()

        if evaluation_results and os.path.exists(evaluation_results) and os.listdir(evaluation_results):
            logger.debug(f"Promoting evaluation data from: {evaluation_results}")
            try:
                eval_data = EvaluationResult.load(evaluation_results)
                log_as = "evaluation"
            except:
                logger.error(f"The files at {evaluation_results} doesn't look to have evaluation-compatible artifacts. Logging them as regular artifacts")
                log_as = "artifacts"
            
            if log_as == "evaluation":
                client.log_batch(
                    run_id=parent_run_id,
                    metrics=[Metric(key, value, timestamp=int(time.time() * 1000), step=0) for key, value in eval_data.metrics.items()]
                )
            else:
                client.log_artifact(
                    run_id=parent_run_id,
                    local_path=evaluation_results
                )

        if include_all_tracking:
            logger.debug(f"Promoting children tracking information")
            for child_run in client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'"):
                logger.debug(f"Promoting child run id: {child_run.info.run_id}")
                logger.debug(f"Promoting parameters: {len(child_run.data.params)} params in total")
                client.log_batch(
                    run_id=parent_run_id, 
                    params=[Param(key, value) for key,value in child_run.data.params.items()]
                )

                for metric_name in child_run.data.metrics.keys():
                    logger.debug(f"Promoting metric: {metric_name}")
                    metric_history = client.get_metric_history(
                        run_id=child_run.info.run_id,
                        key=metric_name
                    )
                    client.log_batch(
                        run_id=parent_run_id,
                        metrics=metric_history
                    )

                if include_all_artifacts:
                    logger.debug(f"Promoting artifacts from the run")
                    for artifact_info in client.list_artifacts(child_run.info.run_id):
                        if artifact_info.is_dir:
                            with tempfile.TemporaryDirectory() as artifact_path:
                                mlflow.artifacts.download_artifacts(
                                    run_id=parent_run_id,
                                    artifact_path=artifact_info.path,
                                    dst_path=artifact_path
                                )
                                client.log_artifacts(
                                    run_id=parent_run_id,
                                    local_dir=artifact_path
                                )
                        else:
                            with tempfile.TemporaryDirectory() as artifact_path:
                                mlflow.artifacts.download_artifacts(
                                    run_id=parent_run_id,
                                    artifact_path=artifact_info.path,
                                    dst_path=artifact_path
                                )
                                client.log_artifact(
                                    run_id=parent_run_id,
                                    local_path=artifact_path
                                )
            
        