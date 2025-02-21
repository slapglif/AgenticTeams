"""Phase 2 diagnostic tests for ATR Framework."""

import os
import json
import time
import uuid
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from web3 import Web3
from eth_tester import EthereumTester, PyEVMBackend
from solcx import compile_source, install_solc

# Add blockchain directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from blockchain.atr_testnet import ATRTestnet

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_diagnostic.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_temporal_constraints(atr_testnet):
    """Test the enforcement of temporal constraints for work unit assignments."""
    logger.info("Starting temporal constraint tests...")
    
    # Register test agent
    agent_type = "TestAgent"
    capabilities = ["test_capability"]
    agent_uuid = atr_testnet.register_agent_onchain(agent_type, capabilities)
    logger.info(f"Registered agent with UUID {agent_uuid}")
    
    # Test 1: Future Start Time
    logger.info("Test 1: Future Start Time")
    future_time = int((datetime.now() + timedelta(hours=1)).timestamp())
    test_work_unit = {
        "name": "Future Work Unit",
        "description": "Test work unit with future start time",
        "version": "1.0",
        "inputs": json.dumps(["input1", "input2"]),
        "outputs": json.dumps(["output1", "output2"]),
        "requirements": json.dumps({
            "agent_type": "TestAgent",
            "capabilities": ["test_capability"]
        }),
        "temporal_constraints": json.dumps({
            "start_time": future_time,
            "deadline": future_time + 3600,
            "max_duration": 3600
        }),
        "data_access_controls": json.dumps({
            "allowed_sources": ["source1"],
            "allowed_queries": ["query1"]
        }),
        "payment_token": "0x0000000000000000000000000000000000000000",
        "payment_amount": Web3.to_wei(1, 'ether')
    }
    
    work_unit_id = atr_testnet.define_work_unit_onchain(
        name=test_work_unit["name"],
        description=test_work_unit["description"],
        version=test_work_unit["version"],
        inputs=test_work_unit["inputs"],
        outputs=test_work_unit["outputs"],
        requirements=test_work_unit["requirements"],
        temporal_constraints=test_work_unit["temporal_constraints"],
        data_access_controls=test_work_unit["data_access_controls"],
        payment_token=test_work_unit["payment_token"],
        payment_amount=test_work_unit["payment_amount"]
    )
    try:
        atr_testnet.assign_agent_to_work_unit_onchain(work_unit_id, agent_uuid)
        logger.error("Test 1 failed: Assignment should not be allowed before start time")
    except Exception as e:
        if "Work unit has not started yet" in str(e):
            logger.info("[PASS] Assignment correctly rejected due to future start time")
        else:
            logger.error(f"Test 1 failed with unexpected error: {str(e)}")
    
    # Test 2: Past Deadline
    logger.info("Test 2: Past Deadline")
    past_time = int((datetime.now() - timedelta(hours=2)).timestamp())
    test_work_unit["temporal_constraints"] = json.dumps({
        "start_time": past_time,
        "deadline": past_time + 3600,
        "max_duration": 3600
    })
    
    work_unit_id = atr_testnet.define_work_unit_onchain(
        name=test_work_unit["name"],
        description=test_work_unit["description"],
        version=test_work_unit["version"],
        inputs=test_work_unit["inputs"],
        outputs=test_work_unit["outputs"],
        requirements=test_work_unit["requirements"],
        temporal_constraints=test_work_unit["temporal_constraints"],
        data_access_controls=test_work_unit["data_access_controls"],
        payment_token=test_work_unit["payment_token"],
        payment_amount=test_work_unit["payment_amount"]
    )
    try:
        atr_testnet.assign_agent_to_work_unit_onchain(work_unit_id, agent_uuid)
        logger.error("Test 2 failed: Assignment should not be allowed after deadline")
    except Exception as e:
        if "Work unit deadline has passed" in str(e):
            logger.info("[PASS] Assignment correctly rejected due to past deadline")
        else:
            logger.error(f"Test 2 failed with unexpected error: {str(e)}")
    
    # Test 3: Valid Temporal Constraints
    logger.info("Test 3: Valid Temporal Constraints")
    current_time = int(datetime.now().timestamp())
    test_work_unit["temporal_constraints"] = json.dumps({
        "start_time": current_time - 1800,
        "deadline": current_time + 1800,
        "max_duration": 3600
    })
    
    work_unit_id = atr_testnet.define_work_unit_onchain(
        name=test_work_unit["name"],
        description=test_work_unit["description"],
        version=test_work_unit["version"],
        inputs=test_work_unit["inputs"],
        outputs=test_work_unit["outputs"],
        requirements=test_work_unit["requirements"],
        temporal_constraints=test_work_unit["temporal_constraints"],
        data_access_controls=test_work_unit["data_access_controls"],
        payment_token=test_work_unit["payment_token"],
        payment_amount=test_work_unit["payment_amount"]
    )
    try:
        atr_testnet.assign_agent_to_work_unit_onchain(work_unit_id, agent_uuid)
        logger.info("[PASS] Assignment successful with valid temporal constraints")
    except Exception as e:
        logger.error(f"Test 3 failed: {str(e)}")

def test_agent_type_compatibility(atr_testnet):
    """Test the enforcement of agent type compatibility for work unit assignments."""
    logger.info("Starting agent type compatibility tests...")
    
    # Register two test agents with different types
    agent1_uuid = atr_testnet.register_agent_onchain("TestAgent", ["test_capability"])
    agent2_uuid = atr_testnet.register_agent_onchain("DifferentAgent", ["test_capability"])
    logger.info(f"Registered compatible agent with UUID {agent1_uuid}")
    logger.info(f"Registered incompatible agent with UUID {agent2_uuid}")
    
    # Create work unit requiring TestAgent type
    current_time = int(datetime.now().timestamp())
    test_work_unit = {
        "name": "Compatibility Test Work Unit",
        "description": "Test work unit for agent type compatibility",
        "version": "1.0",
        "inputs": json.dumps(["input1"]),
        "outputs": json.dumps(["output1"]),
        "requirements": json.dumps({
            "agent_type": "TestAgent",
            "capabilities": ["test_capability"]
        }),
        "temporal_constraints": json.dumps({
            "start_time": current_time - 1800,
            "deadline": current_time + 1800,
            "max_duration": 3600
        }),
        "data_access_controls": json.dumps({
            "allowed_sources": ["source1"],
            "allowed_queries": ["query1"]
        }),
        "payment_token": "0x0000000000000000000000000000000000000000",
        "payment_amount": Web3.to_wei(1, 'ether')
    }
    
    work_unit_id = atr_testnet.define_work_unit_onchain(
        name=test_work_unit["name"],
        description=test_work_unit["description"],
        version=test_work_unit["version"],
        inputs=test_work_unit["inputs"],
        outputs=test_work_unit["outputs"],
        requirements=test_work_unit["requirements"],
        temporal_constraints=test_work_unit["temporal_constraints"],
        data_access_controls=test_work_unit["data_access_controls"],
        payment_token=test_work_unit["payment_token"],
        payment_amount=test_work_unit["payment_amount"]
    )
    
    # Test 1: Compatible Agent
    logger.info("Test 1: Compatible Agent")
    try:
        atr_testnet.assign_agent_to_work_unit_onchain(work_unit_id, agent1_uuid)
        logger.info("[PASS] Assignment successful with compatible agent type")
    except Exception as e:
        logger.error(f"Test 1 failed: {str(e)}")
    
    # Test 2: Incompatible Agent
    logger.info("Test 2: Incompatible Agent")
    # Create a new work unit for Test 2
    work_unit_id = atr_testnet.define_work_unit_onchain(
        name=test_work_unit["name"],
        description=test_work_unit["description"],
        version=test_work_unit["version"],
        inputs=test_work_unit["inputs"],
        outputs=test_work_unit["outputs"],
        requirements=test_work_unit["requirements"],
        temporal_constraints=test_work_unit["temporal_constraints"],
        data_access_controls=test_work_unit["data_access_controls"],
        payment_token=test_work_unit["payment_token"],
        payment_amount=test_work_unit["payment_amount"]
    )
    try:
        atr_testnet.assign_agent_to_work_unit_onchain(work_unit_id, agent2_uuid)
        logger.error("Test 2 failed: Assignment should not be allowed with incompatible agent type")
    except Exception as e:
        if "Agent type incompatible" in str(e):
            logger.info("[PASS] Assignment correctly rejected due to incompatible agent type")
        else:
            logger.error(f"Test 2 failed with unexpected error: {str(e)}")

def run_phase2_diagnostics():
    """Run all Phase 2 diagnostic tests."""
    logger.info("Starting Phase 2 diagnostic tests...")
    
    try:
        # Initialize ATR Framework
        atr_testnet = ATRTestnet()
        
        # Deploy framework contracts
        if not atr_testnet.deploy_framework():
            raise Exception("Failed to deploy ATR Framework contracts")
        logger.info("ATR Framework contracts deployed successfully")
        
        # Run tests
        test_temporal_constraints(atr_testnet)
        test_agent_type_compatibility(atr_testnet)
        
        logger.info("All Phase 2 diagnostic tests completed successfully")
        
    except Exception as e:
        logger.error(f"Phase 2 diagnostic tests failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_phase2_diagnostics() 