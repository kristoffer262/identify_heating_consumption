"""
Main script for the heating consumption identification system.
Orchestrates all agents to process data and identify heating consumption.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# Import agents
from agents.data_agent import DataAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.feature_agent import FeatureAgent
from agents.heating_detection_agent import HeatingDetectionAgent
from agents.modeling_agent import ModelingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.visualization_agent import VisualizationAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_agent_config() -> Dict[str, Any]:
    """Load agent configuration from agents.yaml."""
    config_path = Path("agents.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return {}


def main():
    """Main execution function."""
    logger.info("Starting heating consumption identification system")

    # Load configuration
    config = load_agent_config()
    logger.info(f"Loaded configuration for agents: {config.get('agents', [])}")

    # Initialize agents
    agents = {
        'data_agent': DataAgent(),
        'preprocessing_agent': PreprocessingAgent(),
        'feature_agent': FeatureAgent(),
        'heating_detection_agent': HeatingDetectionAgent(),
        'modeling_agent': ModelingAgent(),
        'evaluation_agent': EvaluationAgent(),
        'visualization_agent': VisualizationAgent()
    }

    # Pipeline execution
    results = {}

    try:
        # 1. Data Loading
        logger.info("Step 1: Loading data")
        data_agent = agents['data_agent']
        raw_data = data_agent.run()
        results['raw_data'] = raw_data

        # 2. Preprocessing
        logger.info("Step 2: Preprocessing data")
        preprocessing_agent = agents['preprocessing_agent']
        processed_data = preprocessing_agent.run(raw_data)
        results['processed_data'] = processed_data

        # 3. Feature Engineering
        logger.info("Step 3: Creating features")
        feature_agent = agents['feature_agent']
        featured_data = feature_agent.run(processed_data)
        results['featured_data'] = featured_data

        # 4. Heating Detection
        logger.info("Step 4: Detecting heating periods")
        heating_agent = agents['heating_detection_agent']
        heating_data = heating_agent.run(featured_data)
        results['heating_data'] = heating_data

        # 5. Modeling
        logger.info("Step 5: Building models")
        modeling_agent = agents['modeling_agent']
        modeling_results = modeling_agent.run(heating_data)
        results['modeling_results'] = modeling_results

        # 6. Evaluation
        logger.info("Step 6: Evaluating models")
        evaluation_agent = agents['evaluation_agent']
        evaluation_results = evaluation_agent.run(modeling_results, heating_data)
        results['evaluation_results'] = evaluation_results

        # 7. Visualization
        logger.info("Step 7: Creating visualizations")
        visualization_agent = agents['visualization_agent']
        visualization_results = visualization_agent.run(heating_data, evaluation_results)
        results['visualization_results'] = visualization_results

        logger.info("Pipeline completed successfully!")
        logger.info(f"Generated {len(visualization_results)} visualizations")

        # Print summary
        print("\n" + "="*50)
        print("HEATING CONSUMPTION IDENTIFICATION - SUMMARY")
        print("="*50)

        if modeling_results and 'best_model' in modeling_results:
            print(f"Best Model: {modeling_results['best_model']}")

        if evaluation_results and 'model_insights' in evaluation_results:
            insights = evaluation_results['model_insights']
            if 'data_stats' in insights:
                stats = insights['data_stats']
                print(f"Total Samples: {stats.get('total_samples', 'N/A')}")
                print(f"Date Range: {stats.get('date_range', 'N/A')}")
                print(f"Heating Periods Detected: {stats.get('heating_periods', 'N/A')}")

        if visualization_results:
            print(f"Visualizations Created: {len(visualization_results)}")
            print("Plot files saved to 'plots/' directory")

        print("="*50)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
