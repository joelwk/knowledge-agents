from literalai import LiteralClient
import logging
from typing import Dict, Optional, Any, Union, List, Callable
import os
from functools import wraps
import time
import asyncio
import traceback
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationRule:
    """Rule for automated evaluation of metrics."""
    name: str
    condition: Callable[[float], bool]
    threshold: float
    action: Callable[[], None]

class MonitoredModelOps:
    """Wrapper class for monitoring model operations using Literal AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the monitoring wrapper with Literal AI client."""
        self.api_key = api_key or os.getenv("LITERAL_API_KEY")
        self.evaluation_rules: List[EvaluationRule] = []
        self.current_thread_id: Optional[str] = None
        self.prompt_cache: Dict[str, Any] = {}  # Cache for prompt templates
        if not self.api_key:
            logger.warning("No Literal AI API key provided. Monitoring will be disabled.")
            self.enabled = False
            return
            
        try:
            self.client = LiteralClient(api_key=self.api_key)
            self.client.instrument_openai()
            self.enabled = True
            # Configure default rules
            self.configure_default_rules()
            logger.info("Literal AI monitoring initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Literal AI monitoring: {e}")
            self.enabled = False
            
    def create_thread(self, query: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """Create a monitored thread for query execution.
        
        Args:
            query: The query being executed
            metadata: Optional metadata for the thread
            
        Returns:
            The thread ID if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            # End current thread if it exists
            if self.current_thread_id:
                self.end_thread(self.current_thread_id)
            
            # Create thread with metadata
            thread_metadata = {
                "query_type": "knowledge_analysis",
                "query_length": len(query),
                "start_time": time.time(),
                "status": "started",
                "query": query[:1000],  # Truncate long queries
                "type": "AI",  # Move type to metadata
                **(metadata or {})
            }
            
            # Create thread using the API directly
            thread = self.client.api.create_thread(
                metadata=thread_metadata,
                tags=["thread", "knowledge_agent"]
            )
            if not thread or not hasattr(thread, 'id'):
                logger.error("Failed to create thread: No thread ID returned")
                return None
                
            # Store current thread ID
            self.current_thread_id = thread.id
            
            # Log thread creation
            try:
                self.client.api.create_score(
                    name="thread_start",
                    type="AI",
                    value=1.0,
                    comment=f"Thread started with query: {query[:100]}{'...' if len(query) > 100 else ''}",
                    step_id=thread.id,
                    tags=["thread", "lifecycle"]
                )
                
                # Update thread status
                self.client.api.update_thread(
                    id=thread.id,
                    metadata={
                        **thread_metadata,
                        "status": "running"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log thread start: {e}")
            
            logger.info(f"Created monitoring thread with ID: {thread.id}")
            return thread.id
            
        except Exception as e:
            logger.error(f"Failed to create monitoring thread: {e}")
            return None
            
    def end_thread(self, thread_id: str) -> None:
        """End a monitoring thread.
        
        Args:
            thread_id: The ID of the thread to end
        """
        if not self.enabled:
            return
            
        try:
            # Get thread metadata
            thread = self.client.api.get_thread(thread_id)
            start_time = float(thread.metadata.get('start_time', time.time()))
            duration = time.time() - start_time
            
            # Update thread status
            self.client.api.update_thread(
                id=thread_id,
                metadata={
                    **thread.metadata,
                    "status": "completed",
                    "end_time": time.time(),
                    "duration": duration
                }
            )
            
            # Log thread completion
            self.client.api.create_score(
                name="thread_end",
                type="AI",
                value=1.0,
                comment=f"Thread completed after {duration:.2f}s",
                step_id=thread_id,
                tags=["thread", "lifecycle"]
            )
            
            # Clear current thread if it matches
            if self.current_thread_id == thread_id:
                self.current_thread_id = None
                
        except Exception as e:
            logger.error(f"Failed to end thread: {e}")
            
    def update_thread_status(self, thread_id: str, status: str, additional_metadata: Optional[Dict] = None) -> None:
        """Update thread status and metadata.
        
        Args:
            thread_id: The ID of the thread to update
            status: New status for the thread
            additional_metadata: Optional additional metadata to update
        """
        if not self.enabled or not thread_id:
            return
            
        try:
            thread = self.client.api.get_thread(thread_id)
            self.client.api.update_thread(
                id=thread_id,
                metadata={
                    **thread.metadata,
                    "status": status,
                    "last_updated": time.time(),
                    **(additional_metadata or {})
                }
            )
        except Exception as e:
            logger.error(f"Failed to update thread status: {e}")
            
    def monitor_step(self, name: str, step_type: str, thread_id: Optional[str] = None):
        """Decorator for monitoring individual steps."""
        # Use current thread if none provided
        if thread_id is None:
            thread_id = self.current_thread_id
            
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled or not thread_id:
                    return await func(*args, **kwargs)
                    
                try:
                    # Update thread status for step start
                    self.update_thread_status(
                        thread_id,
                        f"running_{name}",
                        {"current_step": name}
                    )
                    
                    with self.client.step(
                        name=name,
                        type=step_type,
                        thread_id=thread_id
                    ) as step:
                        # Initialize metadata and tracing
                        step_metadata = {
                            "step": name,
                            "type": step_type,
                            "provider": kwargs.get('provider', {}).value if kwargs.get('provider') else "unknown"
                        }
                        trace_id = f"{thread_id}_{name}"
                        start_time = time.time()
                        
                        # Set initial metadata
                        step.metadata = step_metadata
                        
                        try:
                            # Start trace
                            self.client.api.create_score(
                                name=f"trace_{trace_id}_start",
                                type="AI",
                                value=1.0,
                                comment=f"Starting {name} with {step_metadata['provider']} provider",
                                tags=[trace_id, "trace_start", step_type],
                                step_id=step.id
                            )
                            
                            # Execute function
                            result = await func(*args, **kwargs)
                            
                            # Calculate metrics
                            duration = time.time() - start_time
                            
                            # Update metadata
                            step_metadata.update({
                                "status": "success",
                                "duration_str": f"{duration:.2f}s"
                            })
                            step.metadata = step_metadata
                            
                            # Prepare numeric metrics
                            metrics = {
                                "duration": duration,
                                "success_rate": 1.0
                            }
                            
                            # Add result-specific metrics
                            if isinstance(result, dict):
                                for k, v in result.get('metrics', {}).items():
                                    if isinstance(v, (int, float)):
                                        metrics[k] = float(v)
                            elif isinstance(result, tuple) and len(result) == 2:
                                chunks, summary = result
                                if chunks:
                                    metrics["chunks_generated"] = float(len(chunks))
                                    avg_relevance = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
                                    metrics["average_relevance"] = float(avg_relevance)
                                    metrics["confidence_score"] = float(avg_relevance)
                                if summary:
                                    metrics["summary_length"] = float(len(summary))
                            
                            # Log and evaluate metrics
                            self.log_metrics(metrics, step_id=step.id, metadata=step_metadata)
                            self.evaluate_metrics(metrics, step_id=step.id)
                            
                            # Log success trace
                            self.client.api.create_score(
                                name=f"trace_{trace_id}_success",
                                type="AI",
                                value=1.0,
                                comment=f"Step completed successfully in {duration:.2f}s",
                                tags=[trace_id, "trace_success", step_type],
                                step_id=step.id
                            )
                            
                            # Log experiment metrics
                            self.log_knowledge_agent_metrics(
                                query=kwargs.get('query', ''),
                                summary=summary if isinstance(result, tuple) and len(result) == 2 else '',
                                model_metrics={
                                    step_metadata['provider']: metrics
                                },
                                analysis_metrics={
                                    k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))
                                },
                                step_id=step.id
                            )
                            
                            # Update thread status for success
                            self.update_thread_status(
                                thread_id,
                                f"completed_{name}",
                                {
                                    "current_step": name,
                                    "step_status": "success",
                                    "duration": duration,
                                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                                }
                            )
                            
                            return result
                            
                        except Exception as e:
                            duration = time.time() - start_time
                            
                            # Update thread status for error
                            self.update_thread_status(
                                thread_id,
                                f"error_{name}",
                                {
                                    "current_step": name,
                                    "step_status": "error",
                                    "error_type": type(e).__name__,
                                    "error_message": str(e),
                                    "duration": duration
                                }
                            )
                            
                            # Update metadata for error
                            step_metadata.update({
                                "status": "error",
                                "error_type": type(e).__name__,
                                "duration_str": f"{duration:.2f}s"
                            })
                            step.metadata = step_metadata
                            
                            # Prepare error metrics
                            metrics = {
                                "duration": float(duration),
                                "success_rate": 0.0,
                                "error_rate": 1.0
                            }
                            
                            # Log and evaluate metrics
                            self.log_metrics(metrics, step_id=step.id, metadata=step_metadata)
                            self.evaluate_metrics(metrics, step_id=step.id)
                            
                            # Log detailed error
                            self.log_error(e, step.id)
                            
                            # Log error trace
                            self.client.api.create_score(
                                name=f"trace_{trace_id}_error",
                                type="AI",
                                value=0.0,
                                comment=f"Step failed after {duration:.2f}s: {str(e)}",
                                tags=[trace_id, "trace_error", step_type],
                                step_id=step.id
                            )
                            
                            # Log experiment metrics for failure
                            self.log_knowledge_agent_metrics(
                                query=kwargs.get('query', ''),
                                summary='',
                                model_metrics={
                                    step_metadata['provider']: metrics
                                },
                                analysis_metrics={
                                    k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))
                                },
                                step_id=step.id
                            )
                            
                            raise
                            
                except Exception as e:
                    logger.error(f"Failed to monitor step: {e}")
                    return await func(*args, **kwargs)
                        
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled or not thread_id:
                    return func(*args, **kwargs)
                    
                try:
                    with self.client.step(
                        name=name,
                        type=step_type,
                        thread_id=thread_id
                    ) as step:
                        # Initialize metadata and tracing
                        step_metadata = {
                            "step": name,
                            "type": step_type,
                            "provider": kwargs.get('provider', {}).value if kwargs.get('provider') else "unknown"
                        }
                        trace_id = f"{thread_id}_{name}"
                        start_time = time.time()
                        
                        # Set initial metadata
                        step.metadata = step_metadata
                        
                        try:
                            # Start trace
                            self.client.api.create_score(
                                name=f"trace_{trace_id}_start",
                                type="AI",
                                value=1.0,
                                comment=f"Starting {name} with {step_metadata['provider']} provider",
                                tags=[trace_id, "trace_start", step_type],
                                step_id=step.id
                            )
                            
                            # Execute function
                            result = func(*args, **kwargs)
                            
                            # Calculate metrics
                            duration = time.time() - start_time
                            
                            # Update metadata
                            step_metadata.update({
                                "status": "success",
                                "duration_str": f"{duration:.2f}s"
                            })
                            step.metadata = step_metadata
                            
                            # Prepare numeric metrics
                            metrics = {
                                "duration": float(duration),
                                "success_rate": 1.0
                            }
                            
                            # Add result-specific metrics
                            if isinstance(result, dict):
                                for k, v in result.get('metrics', {}).items():
                                    if isinstance(v, (int, float)):
                                        metrics[k] = float(v)
                            elif isinstance(result, tuple) and len(result) == 2:
                                chunks, summary = result
                                if chunks:
                                    metrics["chunks_generated"] = float(len(chunks))
                                    avg_relevance = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
                                    metrics["average_relevance"] = float(avg_relevance)
                                    metrics["confidence_score"] = float(avg_relevance)
                                if summary:
                                    metrics["summary_length"] = float(len(summary))
                            
                            # Log and evaluate metrics
                            self.log_metrics(metrics, step_id=step.id, metadata=step_metadata)
                            self.evaluate_metrics(metrics, step_id=step.id)
                            
                            # Log success trace
                            self.client.api.create_score(
                                name=f"trace_{trace_id}_success",
                                type="AI",
                                value=1.0,
                                comment=f"Step completed successfully in {duration:.2f}s",
                                tags=[trace_id, "trace_success", step_type],
                                step_id=step.id
                            )
                            
                            # Log experiment metrics
                            self.log_knowledge_agent_metrics(
                                query=kwargs.get('query', ''),
                                summary=summary if isinstance(result, tuple) and len(result) == 2 else '',
                                model_metrics={
                                    step_metadata['provider']: metrics
                                },
                                analysis_metrics={
                                    k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))
                                },
                                step_id=step.id
                            )
                            
                            # Update thread status for success
                            self.update_thread_status(
                                thread_id,
                                f"completed_{name}",
                                {
                                    "current_step": name,
                                    "step_status": "success",
                                    "duration": duration,
                                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                                }
                            )
                            
                            return result
                            
                        except Exception as e:
                            duration = time.time() - start_time
                            
                            # Update thread status for error
                            self.update_thread_status(
                                thread_id,
                                f"error_{name}",
                                {
                                    "current_step": name,
                                    "step_status": "error",
                                    "error_type": type(e).__name__,
                                    "error_message": str(e),
                                    "duration": duration
                                }
                            )
                            
                            # Update metadata for error
                            step_metadata.update({
                                "status": "error",
                                "error_type": type(e).__name__,
                                "duration_str": f"{duration:.2f}s"
                            })
                            step.metadata = step_metadata
                            
                            # Prepare error metrics
                            metrics = {
                                "duration": float(duration),
                                "success_rate": 0.0,
                                "error_rate": 1.0
                            }
                            
                            # Log and evaluate metrics
                            self.log_metrics(metrics, step_id=step.id, metadata=step_metadata)
                            self.evaluate_metrics(metrics, step_id=step.id)
                            
                            # Log detailed error
                            self.log_error(e, step.id)
                            
                            # Log error trace
                            self.client.api.create_score(
                                name=f"trace_{trace_id}_error",
                                type="AI",
                                value=0.0,
                                comment=f"Step failed after {duration:.2f}s: {str(e)}",
                                tags=[trace_id, "trace_error", step_type],
                                step_id=step.id
                            )
                            
                            # Log experiment metrics for failure
                            self.log_knowledge_agent_metrics(
                                query=kwargs.get('query', ''),
                                summary='',
                                model_metrics={
                                    step_metadata['provider']: metrics
                                },
                                analysis_metrics={
                                    k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))
                                },
                                step_id=step.id
                            )
                            
                            raise
                            
                except Exception as e:
                    logger.error(f"Failed to monitor step: {e}")
                    return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics using scores.
        
        Args:
            metrics: Dictionary of numeric metrics
            step_id: Optional step ID to associate with the scores
            metadata: Optional metadata to include in score comments
        """
        if not self.enabled:
            return
            
        try:
            for metric_name, value in metrics.items():
                if not step_id:
                    logger.warning(f"Cannot log metric {metric_name} without step_id")
                    continue
                
                if not isinstance(value, (int, float)):
                    logger.warning(f"Skipping metric {metric_name} - value must be numeric")
                    continue
                
                comment = None
                if metadata:
                    comment = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                    
                self.client.api.create_score(
                    name=metric_name,
                    type="AI",
                    value=float(value),
                    step_id=step_id,
                    comment=comment,
                    tags=["metric"]
                )
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            
    def log_input(self, input_data: str, step_id: Optional[str] = None) -> None:
        """Log input data using score."""
        if not self.enabled or not step_id:
            return
            
        try:
            self.client.api.create_score(
                name="input_length",
                type="AI",
                value=float(len(input_data)),
                step_id=step_id,
                tags=["input", "length"]
            )
        except Exception as e:
            logger.error(f"Failed to log input: {e}")
            
    def log_output(self, output_data: str, step_id: Optional[str] = None) -> None:
        """Log output data using score."""
        if not self.enabled or not step_id:
            return
            
        try:
            self.client.api.create_score(
                name="output_length",
                type="AI",
                value=float(len(output_data)),
                step_id=step_id,
                tags=["output", "length"]
            )
        except Exception as e:
            logger.error(f"Failed to log output: {e}")

    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], step_id: Optional[str] = None) -> None:
        """Log model-specific metrics."""
        if not self.enabled or not step_id:
            return
            
        try:
            scores = [
                {
                    'name': metric_name,
                    'value': value,
                    'comment': f'{metric_name.replace("_", " ").title()} for {model_name}',
                    'tags': ['model', model_name, metric_name]
                }
                for metric_name, value in metrics.items()
            ]
            self.log_experiment(f"{model_name}_metrics", "", "", scores, step_id)
        except Exception as e:
            logger.error(f"Failed to log model metrics: {e}")

    def log_analysis_metrics(self, analysis_type: str, metrics: Dict[str, float], step_id: Optional[str] = None) -> None:
        """Log analysis-specific metrics."""
        if not self.enabled or not step_id:
            return
            
        try:
            scores = [
                {
                    'name': metric_name,
                    'value': value,
                    'comment': f'{metric_name.replace("_", " ").title()} for {analysis_type}',
                    'tags': ['analysis', analysis_type, metric_name]
                }
                for metric_name, value in metrics.items()
            ]
            self.log_experiment(f"{analysis_type}_analysis", "", "", scores, step_id)
        except Exception as e:
            logger.error(f"Failed to log analysis metrics: {e}")

    def log_experiment(self, experiment_name: str, input_data: str, output_data: str, scores: List[Dict[str, Any]], step_id: Optional[str] = None) -> None:
        """Log an experiment with input, output, and scores.
        
        Args:
            experiment_name: Name of the experiment
            input_data: Input data or query
            output_data: Output data or results
            scores: List of score dictionaries with format:
                   {
                       'name': str,  # Name of the metric
                       'value': float,  # Numeric value
                       'comment': str,  # Optional description
                       'tags': List[str]  # Optional tags for categorization
                   }
            step_id: ID of the step this experiment is part of
        """
        if not self.enabled or not step_id:
            return
            
        try:
            # Log input/output if provided
            if input_data:
                self.log_input(input_data, step_id)
            if output_data:
                self.log_output(output_data, step_id)
            
            # Log all scores
            for score in scores:
                if not isinstance(score.get('value'), (int, float)):
                    logger.warning(f"Skipping score {score['name']} - value must be numeric")
                    continue
                    
                self.client.api.create_score(
                    name=f"experiment_{experiment_name}_{score['name']}",
                    type="AI",
                    value=float(score['value']),
                    comment=score.get('comment', ''),
                    step_id=step_id,
                    tags=["experiment", experiment_name] + score.get('tags', [])
                )
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")

    def log_knowledge_agent_metrics(self, 
                                  query: str,
                                  summary: str,
                                  model_metrics: Dict[str, Dict[str, float]],
                                  analysis_metrics: Dict[str, float],
                                  step_id: Optional[str] = None) -> None:
        """Log comprehensive metrics for knowledge agent execution."""
        if not self.enabled or not step_id:
            return
            
        try:
            # Log query and summary lengths
            numeric_metrics = {
                'query_length': float(len(query)),
                'summary_length': float(len(summary))
            }
            
            metadata = {
                'experiment_type': 'knowledge_agent_run',
                'has_query': bool(query),
                'has_summary': bool(summary)
            }
            
            self.log_metrics(numeric_metrics, step_id=step_id, metadata=metadata)
            
            # Log model-specific metrics
            for model_name, metrics in model_metrics.items():
                numeric_model_metrics = {
                    k: v for k, v in metrics.items() 
                    if isinstance(v, (int, float))
                }
                if numeric_model_metrics:
                    self.log_metrics(
                        numeric_model_metrics,
                        step_id=step_id,
                        metadata={'model': model_name}
                    )
            
            # Log analysis metrics
            numeric_analysis_metrics = {
                k: v for k, v in analysis_metrics.items()
                if isinstance(v, (int, float))
            }
            if numeric_analysis_metrics:
                self.log_metrics(
                    numeric_analysis_metrics,
                    step_id=step_id,
                    metadata={'metric_type': 'analysis'}
                )
                
        except Exception as e:
            logger.error(f"Failed to log knowledge agent metrics: {e}")

    def log_user_feedback(self, feedback: int, comment: str, step_id: Optional[str] = None) -> None:
        """Log user feedback as a score."""
        if not self.enabled or not step_id:
            return
            
        try:
            self.client.api.create_score(
                name="user-feedback",
                type="HUMAN",
                value=feedback,
                comment=comment,
                step_id=step_id,
                tags=["feedback"]
            )
        except Exception as e:
            logger.error(f"Failed to log user feedback: {e}")

    def add_evaluation_rule(self, rule: EvaluationRule) -> None:
        """Add an automated evaluation rule."""
        self.evaluation_rules.append(rule)

    def evaluate_metrics(self, metrics: Dict[str, Union[int, float]], step_id: Optional[str] = None) -> None:
        """Evaluate metrics against defined rules."""
        if not self.enabled:
            return

        for rule in self.evaluation_rules:
            if rule.name in metrics:
                value = float(metrics[rule.name])
                if rule.condition(value):
                    try:
                        rule.action(step_id)
                    except Exception as e:
                        logger.error(f"Failed to execute rule action: {e}")

    def log_error(self, error: Exception, step_id: Optional[str] = None) -> None:
        """Log detailed error information including stack trace."""
        if not self.enabled or not step_id:
            return

        try:
            stack_trace = ''.join(traceback.format_tb(error.__traceback__))
            error_message = (
                f"Error Type: {type(error).__name__}\n"
                f"Error Message: {str(error)}\n"
                f"Stack Trace:\n{stack_trace}"
            )
            
            self.client.api.create_score(
                name="error_details",
                type="AI",
                value=1.0,
                comment=error_message,
                step_id=step_id,
                tags=["error", type(error).__name__]
            )
        except Exception as e:
            logger.error(f"Failed to log error details: {e}")

    def configure_default_rules(self) -> None:
        """Configure default evaluation rules for monitoring."""
        if not self.enabled:
            return

        # Rule for slow operations
        def alert_slow_operation(step_id: Optional[str] = None):
            if not step_id:
                logger.warning("Operation exceeded duration threshold")
                return
                
            try:
                self.client.api.create_score(
                    name="alert_slow_operation",
                    type="AI",
                    value=0.0,
                    comment="Operation exceeded duration threshold of 5.0 seconds",
                    tags=["alert", "performance"],
                    step_id=step_id
                )
            except Exception as e:
                logger.error(f"Failed to create alert score: {e}")

        self.add_evaluation_rule(EvaluationRule(
            name="duration",
            condition=lambda x: x > 5.0,  # 5 seconds threshold
            threshold=5.0,
            action=alert_slow_operation
        ))

        # Rule for high error rate
        def alert_high_error_rate(step_id: Optional[str] = None):
            if not step_id:
                logger.warning("High error rate detected")
                return
                
            try:
                self.client.api.create_score(
                    name="alert_high_error_rate",
                    type="AI",
                    value=0.0,
                    comment="Error rate exceeded threshold of 10%",
                    tags=["alert", "reliability"],
                    step_id=step_id
                )
            except Exception as e:
                logger.error(f"Failed to create alert score: {e}")

        self.add_evaluation_rule(EvaluationRule(
            name="error_rate",
            condition=lambda x: x > 0.1,  # 10% error rate threshold
            threshold=0.1,
            action=alert_high_error_rate
        ))

        # Rule for low confidence scores
        def alert_low_confidence(step_id: Optional[str] = None):
            if not step_id:
                logger.warning("Low confidence score detected")
                return
                
            try:
                self.client.api.create_score(
                    name="alert_low_confidence",
                    type="AI",
                    value=0.0,
                    comment="Confidence score below threshold of 70%",
                    tags=["alert", "quality"],
                    step_id=step_id
                )
            except Exception as e:
                logger.error(f"Failed to create alert score: {e}")

        self.add_evaluation_rule(EvaluationRule(
            name="confidence_score",
            condition=lambda x: x < 0.7,  # 70% confidence threshold
            threshold=0.7,
            action=alert_low_confidence
        ))

        # Rule for low success rate
        def alert_low_success(step_id: Optional[str] = None):
            if not step_id:
                logger.warning("Low success rate detected")
                return
                
            try:
                self.client.api.create_score(
                    name="alert_low_success",
                    type="AI",
                    value=0.0,
                    comment="Success rate below threshold of 90%",
                    tags=["alert", "reliability"],
                    step_id=step_id
                )
            except Exception as e:
                logger.error(f"Failed to create alert score: {e}")

        self.add_evaluation_rule(EvaluationRule(
            name="success",
            condition=lambda x: x < 0.9,  # 90% success threshold
            threshold=0.9,
            action=alert_low_success
        ))

        # Rule for low relevance scores
        def alert_low_relevance(step_id: Optional[str] = None):
            if not step_id:
                logger.warning("Low relevance score detected")
                return
                
            try:
                self.client.api.create_score(
                    name="alert_low_relevance",
                    type="AI",
                    value=0.0,
                    comment="Average relevance score below threshold of 0.6",
                    tags=["alert", "quality"],
                    step_id=step_id
                )
            except Exception as e:
                logger.error(f"Failed to create alert score: {e}")

        self.add_evaluation_rule(EvaluationRule(
            name="average_relevance",
            condition=lambda x: x < 0.6,  # 60% relevance threshold
            threshold=0.6,
            action=alert_low_relevance
        ))

    def get_or_create_prompt(self, 
                           name: str,
                           messages: List[Dict[str, str]],
                           model: str = "gpt-4",
                           provider: str = "openai",
                           settings: Optional[Dict] = None) -> Optional[str]:
        """Get or create a prompt template.
        
        Args:
            name: Name of the prompt template
            messages: List of message dictionaries (role, content)
            model: Model to use (default: gpt-4)
            provider: Provider to use (default: openai)
            settings: Optional model settings
            
        Returns:
            Prompt ID if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            # Check cache first
            if name in self.prompt_cache:
                return self.prompt_cache[name]['id']
                
            # Create prompt template
            prompt = self.client.api.get_or_create_prompt(
                name=name,
                messages=messages,
                model=model,
                provider=provider,
                settings=settings or {}
            )
            
            # Cache the prompt
            self.prompt_cache[name] = prompt
            return prompt.id
            
        except Exception as e:
            logger.error(f"Failed to get/create prompt template: {e}")
            return None
            
    def log_prompt_usage(self, 
                        prompt_id: str,
                        step_id: str,
                        variables: Dict[str, Any],
                        completion: str,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Log prompt usage with variables and completion.
        
        Args:
            prompt_id: ID of the prompt template used
            step_id: ID of the step where prompt was used
            variables: Variables used in the prompt
            completion: Model completion/response
            metrics: Optional metrics about the completion
        """
        if not self.enabled:
            return
            
        try:
            # Create generation record
            generation = self.client.api.create_generation(
                prompt_id=prompt_id,
                step_id=step_id,
                variables=variables,
                completion=completion,
                metadata={
                    "timestamp": time.time(),
                    **(metrics or {})
                }
            )
            
            # Log metrics if provided
            if metrics:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.client.api.create_score(
                            name=f"prompt_{name}",
                            type="AI",
                            value=float(value),
                            step_id=step_id,
                            tags=["prompt", "generation"]
                        )
                        
        except Exception as e:
            logger.error(f"Failed to log prompt usage: {e}")
            
    def monitor_prompt(self, name: str, messages: List[Dict[str, str]], model: str = "gpt-4"):
        """Decorator for monitoring prompt usage.
        
        Args:
            name: Name of the prompt template
            messages: List of message dictionaries
            model: Model to use
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                    
                # Get current step ID from context
                step_id = None
                if hasattr(self.client, 'get_current_step'):
                    step = self.client.get_current_step()
                    if step:
                        step_id = step.id
                
                try:
                    # Get or create prompt template
                    prompt_id = self.get_or_create_prompt(
                        name=name,
                        messages=messages,
                        model=model,
                        provider=kwargs.get('provider', 'openai')
                    )
                    
                    if not prompt_id:
                        return await func(*args, **kwargs)
                    
                    # Execute function
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Extract completion and metrics
                    completion = result
                    metrics = {
                        "duration": duration,
                        "tokens": len(str(result)) // 4  # Rough estimate
                    }
                    
                    if isinstance(result, dict):
                        completion = result.get('completion', str(result))
                        metrics.update({
                            k: v for k, v in result.items()
                            if isinstance(v, (int, float))
                        })
                    
                    # Log prompt usage
                    self.log_prompt_usage(
                        prompt_id=prompt_id,
                        step_id=step_id,
                        variables=kwargs,
                        completion=str(completion),
                        metrics=metrics
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to monitor prompt: {e}")
                    return await func(*args, **kwargs)
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Same implementation but for sync functions
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                # Get current step ID from context
                step_id = None
                if hasattr(self.client, 'get_current_step'):
                    step = self.client.get_current_step()
                    if step:
                        step_id = step.id
                
                try:
                    # Get or create prompt template
                    prompt_id = self.get_or_create_prompt(
                        name=name,
                        messages=messages,
                        model=model,
                        provider=kwargs.get('provider', 'openai')
                    )
                    
                    if not prompt_id:
                        return func(*args, **kwargs)
                    
                    # Execute function
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Extract completion and metrics
                    completion = result
                    metrics = {
                        "duration": duration,
                        "tokens": len(str(result)) // 4  # Rough estimate
                    }
                    
                    if isinstance(result, dict):
                        completion = result.get('completion', str(result))
                        metrics.update({
                            k: v for k, v in result.items()
                            if isinstance(v, (int, float))
                        })
                    
                    # Log prompt usage
                    self.log_prompt_usage(
                        prompt_id=prompt_id,
                        step_id=step_id,
                        variables=kwargs,
                        completion=str(completion),
                        metrics=metrics
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to monitor prompt: {e}")
                    return func(*args, **kwargs)
                    
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

def get_monitored_ops(api_key: Optional[str] = None) -> MonitoredModelOps:
    """Factory function to get monitored operations instance."""
    return MonitoredModelOps(api_key=api_key) 