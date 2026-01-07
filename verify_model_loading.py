
import sys
import os
from loguru import logger
from master_integration_system import MasterIntegrationSystem

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

def verify():
    logger.info("Verifying MasterIntegrationSystem with real models...")
    
    # Initialize with use_ml=True
    system = MasterIntegrationSystem(use_ml=True)
    
    # Check if models are loaded
    if system.lstm_model is not None:
        logger.success("LSTM Model object initialized")
    else:
        logger.error("LSTM Model object IS NONE")
        
    if system.xgboost_model is not None:
        logger.success("XGBoost Model object initialized")
    else:
        logger.error("XGBoost Model object IS NONE")
        
    # Check if we can determine if they loaded weights?
    # The load_models logs info, so we watch the output.
    
    logger.info(f"System use_ml status: {system.use_ml}")
    
    if system.use_ml:
        logger.success("System accepted ML models and kept use_ml=True")
    else:
        logger.error("System disabled use_ml (likely due to loading failure)")

if __name__ == "__main__":
    verify()
