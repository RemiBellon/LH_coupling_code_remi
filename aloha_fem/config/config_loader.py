import os
from config.schema import SimulationConfig

def load_config(yaml_filepath: str) -> SimulationConfig:
    """
    Reads the YAML configuration file and validates it strictly against the schema.
    Returns a robust SimulationConfig object that can be passed safely to the FEM solver.
    """
    if not os.path.exists(yaml_filepath):
        raise FileNotFoundError(f"[!] Configuration file not found at: {yaml_filepath}")

    try:
        # We leverage the elegant from_yaml class method you already built in schema.py
        validated_config = SimulationConfig.from_yaml(yaml_filepath)
        print(f"--- Configuration successfully loaded and validated from {yaml_filepath} ---")
        return validated_config
        
    except TypeError as e:
        raise TypeError(f"[!] Configuration Validation Error. Check your YAML structure against the schema:\n{e}")
    except Exception as e:
        raise ValueError(f"[!] Unexpected error during configuration validation:\n{e}")