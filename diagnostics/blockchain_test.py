"""Diagnostic tests for blockchain integration using Sepolia testnet."""

import logging
import sys
import os
from web3 import Web3, EthereumTesterProvider
from eth_tester import PyEVMBackend
import solcx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('blockchain_test.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_solidity():
    """Install and set up the Solidity compiler."""
    try:
        logger.info("Setting up Solidity compiler...")
        # Install the latest compatible version
        solcx.install_solc('0.8.23')
        # Set as the default version
        solcx.set_solc_version('0.8.23')
        logger.info("[PASS] Solidity compiler setup complete")
        return True
    except Exception as e:
        logger.error(f"Failed to set up Solidity compiler: {str(e)}", exc_info=True)
        return False

def run_diagnostic_tests():
    """Run a series of diagnostic tests for blockchain integration."""
    logger.info("Starting blockchain integration diagnostic tests...")
    
    # First set up the Solidity compiler
    if not setup_solidity():
        return False
    
    try:
        # Initialize provider and Web3
        provider = EthereumTesterProvider(PyEVMBackend())
        w3 = Web3(provider)
        logger.info("Successfully connected to local testnet")
        
        # Test 1: Check connection
        logger.info("\nTest 1: Connection Check")
        assert w3.is_connected(), "Failed to connect to the blockchain"
        logger.info("[PASS] Connection test passed")
        
        # Test 2: Check accounts
        logger.info("\nTest 2: Account Check")
        accounts = w3.eth.accounts
        assert len(accounts) > 0, "No accounts found"
        test_account = accounts[0]
        balance = w3.eth.get_balance(test_account)
        logger.info(f"[PASS] Found {len(accounts)} accounts")
        logger.info(f"[PASS] Test account balance: {Web3.from_wei(balance, 'ether')} ETH")
        
        # Test 3: Transaction Test
        logger.info("\nTest 3: Transaction Test")
        recipient = accounts[1]
        tx_hash = w3.eth.send_transaction({
            'from': test_account,
            'to': recipient,
            'value': Web3.to_wei(1, 'ether')
        })
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt.status == 1, "Transaction failed"
        logger.info("[PASS] Successfully sent 1 ETH transaction")
        
        # Test 4: Smart Contract Deployment Test
        logger.info("\nTest 4: Smart Contract Deployment")
        test_contract = """
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.0;
        
        contract TestContract {
            string public message;
            
            constructor(string memory _message) {
                message = _message;
            }
            
            function setMessage(string memory _message) public {
                message = _message;
            }
        }
        """
        
        # Compile and deploy contract
        compiled_sol = solcx.compile_source(test_contract)
        contract_interface = compiled_sol['<stdin>:TestContract']
        
        Contract = w3.eth.contract(
            abi=contract_interface['abi'],
            bytecode=contract_interface['bin']
        )
        
        tx_hash = Contract.constructor("Hello, Blockchain!").transact({'from': test_account})
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt.status == 1, "Contract deployment failed"
        
        contract_instance = w3.eth.contract(
            address=tx_receipt.contractAddress,
            abi=contract_interface['abi']
        )
        
        message = contract_instance.functions.message().call()
        assert message == "Hello, Blockchain!", "Contract state verification failed"
        logger.info("[PASS] Successfully deployed and interacted with test contract")
        
        logger.info("\nAll diagnostic tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Diagnostic test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_diagnostic_tests()
    sys.exit(0 if success else 1) 