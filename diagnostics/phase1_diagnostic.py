"""Phase 1 diagnostic tests for ATR Framework."""

import os
import json
import logging
import sys
from pathlib import Path

# Add blockchain directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from blockchain.atr_testnet import ATRTestnet
from web3 import Web3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phase1_diagnostic.log')
    ]
)
logger = logging.getLogger(__name__)

def test_agent_registration(atr: ATRTestnet) -> bool:
    """
    Test agent registration functionality.
    
    Args:
        atr: ATRTestnet instance
        
    Returns:
        bool indicating success
    """
    try:
        logger.info("\n=== Testing Agent Registration ===")
        
        # Test data
        agent_id = "testAgent1"
        agent_type = "TestAgent"
        capabilities = ["test_capability"]
        
        # Register agent
        logger.info(f"Registering agent {agent_id}...")
        success = atr.register_agent_onchain(agent_id, agent_type, capabilities)
        if not success:
            logger.error("Failed to register agent")
            return False
        
        # Get agent details
        logger.info("Retrieving agent details...")
        agent_details = atr.get_agent_details_onchain(agent_id)
        
        # Verify agent details
        assert agent_details['agent_id'] == agent_id, "Agent ID mismatch"
        assert agent_details['agent_type'] == agent_type, "Agent type mismatch"
        assert agent_details['capabilities'] == capabilities, "Capabilities mismatch"
        assert agent_details['is_active'] is True, "Agent should be active"
        
        logger.info("Agent details retrieved successfully:")
        logger.info(json.dumps(agent_details, indent=2))
        logger.info("[PASS] Agent registration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Agent registration test failed: {str(e)}")
        return False

def test_work_unit_definition(atr: ATRTestnet) -> bool:
    """
    Test work unit definition functionality.
    
    Args:
        atr: ATRTestnet instance
        
    Returns:
        bool indicating success
    """
    try:
        logger.info("\n=== Testing Work Unit Definition ===")
        
        # Test data
        test_work_unit = {
            'name': "Test Work Unit",
            'description': "A test work unit for ATR Framework",
            'version': "1.0",
            'inputs': json.dumps({
                'test_input': {
                    'type': 'string',
                    'description': 'Test input parameter'
                }
            }),
            'outputs': json.dumps({
                'test_output': {
                    'type': 'string',
                    'description': 'Test output parameter'
                }
            }),
            'requirements': json.dumps({
                'agent_type': 'TestAgent',
                'capabilities': ['test_capability']
            }),
            'temporal_constraints': json.dumps({
                'start_time': 0,
                'deadline': 2**32 - 1,
                'max_duration': 3600
            }),
            'data_access_controls': json.dumps({
                'allowed_sources': ['test_source'],
                'allowed_queries': ['test_query']
            }),
            'payment_token': "0x0000000000000000000000000000000000000000",
            'payment_amount': Web3.to_wei(1, 'ether')
        }
        
        # Define work unit
        logger.info("Defining work unit...")
        work_unit_id = atr.define_work_unit_onchain(
            name=test_work_unit['name'],
            description=test_work_unit['description'],
            version=test_work_unit['version'],
            inputs=test_work_unit['inputs'],
            outputs=test_work_unit['outputs'],
            requirements=test_work_unit['requirements'],
            temporal_constraints=test_work_unit['temporal_constraints'],
            data_access_controls=test_work_unit['data_access_controls'],
            payment_token=test_work_unit['payment_token'],
            payment_amount=test_work_unit['payment_amount']
        )
        
        # Get work unit details
        logger.info(f"Retrieving work unit {work_unit_id} details...")
        work_unit = atr.get_work_unit_onchain(work_unit_id)
        
        # Verify work unit details
        assert work_unit['name'] == test_work_unit['name'], "Name mismatch"
        assert work_unit['description'] == test_work_unit['description'], "Description mismatch"
        assert work_unit['version'] == test_work_unit['version'], "Version mismatch"
        assert work_unit['inputs'] == test_work_unit['inputs'], "Inputs mismatch"
        assert work_unit['outputs'] == test_work_unit['outputs'], "Outputs mismatch"
        assert work_unit['requirements'] == test_work_unit['requirements'], "Requirements mismatch"
        assert work_unit['temporal_constraints'] == test_work_unit['temporal_constraints'], "Temporal constraints mismatch"
        assert work_unit['data_access_controls'] == test_work_unit['data_access_controls'], "Data access controls mismatch"
        assert work_unit['payment_token'] == test_work_unit['payment_token'], "Payment token mismatch"
        assert work_unit['payment_amount'] == test_work_unit['payment_amount'], "Payment amount mismatch"
        assert work_unit['is_active'] is True, "Work unit should be active"
        assert work_unit['assigned_agent_id'] == "", "Work unit should not be assigned"
        assert int(work_unit['status']) == 1, "Work unit status should be CREATED"
        
        logger.info("Work unit details retrieved successfully:")
        logger.info(json.dumps(work_unit, indent=2))
        logger.info("[PASS] Work unit definition test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Work unit definition test failed: {str(e)}")
        return False

def run_phase1_diagnostics():
    """Run all Phase 1 diagnostic tests."""
    try:
        logger.info("Starting Phase 1 diagnostic tests...")
        
        # Initialize ATR Framework
        atr = ATRTestnet()
        
        # Deploy framework contracts
        logger.info("\nDeploying ATR Framework contracts...")
        if not atr.deploy_framework():
            logger.error("Failed to deploy framework")
            return False
        
        # Run tests
        tests_passed = True
        
        # Test 1: Agent Registration
        if not test_agent_registration(atr):
            tests_passed = False
            logger.error("Agent registration test failed")
        
        # Test 2: Work Unit Definition
        if not test_work_unit_definition(atr):
            tests_passed = False
            logger.error("Work unit definition test failed")
        
        if tests_passed:
            logger.info("\n=== All Phase 1 diagnostic tests passed successfully! ===")
        else:
            logger.error("\n=== Some Phase 1 diagnostic tests failed ===")
        
        return tests_passed
        
    except Exception as e:
        logger.error(f"Phase 1 diagnostics failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_phase1_diagnostics()
    sys.exit(0 if success else 1) 