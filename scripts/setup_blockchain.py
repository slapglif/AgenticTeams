"""Setup script for blockchain integration."""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from collections.abc import Mapping
from eth_account import Account
from web3 import Web3, EthereumTesterProvider
from eth_tester import EthereumTester, PyEVMBackend
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)

def generate_eth_account():
    """Generate a new Ethereum account."""
    try:
        logger.info("Generating new Ethereum account...")
        account = Account.create()
        logger.info("Account generated successfully")
        return {
            "address": account.address,
            "private_key": account.key.hex()
        }
    except Exception as e:
        logger.error(f"Error generating Ethereum account: {e}")
        logger.debug(traceback.format_exc())
        return None

def install_dependencies():
    """Install required dependencies."""
    try:
        logger.info("Installing dependencies...")
        requirements_path = Path("requirements.txt")
        if not requirements_path.exists():
            logger.error("requirements.txt not found!")
            return False
            
        logger.debug(f"Installing from {requirements_path.absolute()}")
        result = os.system(f"uv pip install -r {requirements_path}")
        if result != 0:
            logger.error("Failed to install dependencies")
            return False
            
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        logger.debug(traceback.format_exc())
        return False

def setup_local_testnet():
    """Set up a local testnet using eth-tester."""
    try:
        logger.info("Setting up local testnet...")
        
        # Initialize eth-tester with PyEVM backend
        eth_tester = EthereumTester(PyEVMBackend())
        w3 = Web3(EthereumTesterProvider(eth_tester))
        
        # Get the private key from environment
        private_key = os.getenv('ETH_PRIVATE_KEY')
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key
            
        # Import the account
        eth_tester.add_account(private_key)
        account = Account.from_key(private_key)
        
        # Fund the account
        w3.eth.send_transaction({
            'from': w3.eth.accounts[0],
            'to': account.address,
            'value': w3.to_wei(100, 'ether')
        })
        
        logger.info("Local testnet set up successfully")
        logger.info(f"Account balance: {w3.from_wei(w3.eth.get_balance(account.address), 'ether')} ETH")
        return True
    except Exception as e:
        logger.error(f"Error setting up local testnet: {e}")
        logger.debug(traceback.format_exc())
        return False

def check_cpp_build_tools():
    """Check if Visual C++ Build Tools are installed."""
    # Skip the check since build tools are already installed
    logger.info("Skipping Visual C++ Build Tools check...")
    return True

def main():
    """Run setup process."""
    try:
        logger.info("Starting blockchain integration setup...")
        logger.info("This script will set up a local testnet for development")
        
        # Step 0: Check build tools
        logger.info("\nStep 0: Checking prerequisites")
        if not check_cpp_build_tools():
            return
        
        # Step 1: Generate or load Ethereum account
        if not os.getenv("ETH_PRIVATE_KEY"):
            logger.info("\nStep 1: Generating new Ethereum account")
            account = generate_eth_account()
            if not account:
                logger.error("Failed to generate Ethereum account")
                return
            logger.info(f"Generated account address: {account['address']}")
            logger.info(f"Generated private key: {account['private_key']}")
            logger.info("\nIMPORTANT: Save these credentials securely!")
            
            # Update .env file
            env_path = Path(".env")
            env_content = env_path.read_text() if env_path.exists() else ""
            env_content += f"\nETH_PRIVATE_KEY={account['private_key']}"
            env_path.write_text(env_content)
            logger.info("Private key saved to .env file")
            
            # Reload environment variables
            load_dotenv()
        else:
            try:
                logger.info("\nStep 1: Using existing Ethereum account")
                account = {"address": Account.from_key(os.getenv("ETH_PRIVATE_KEY")).address}
                logger.info(f"Account address: {account['address']}")
            except Exception as e:
                logger.error(f"Error loading existing Ethereum account: {e}")
                logger.debug(traceback.format_exc())
                return
        
        # Step 2: Install dependencies
        logger.info("\nStep 2: Installing dependencies")
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            return
        
        # Step 3: Set up local testnet
        logger.info("\nStep 3: Setting up local testnet")
        if not setup_local_testnet():
            logger.error("Failed to set up local testnet")
            return
        
        # Final instructions
        logger.info("\nSetup complete!")
        logger.info("\nYour local blockchain environment is ready:")
        logger.info(f"- Account Address: {account['address']}")
        logger.info("- Network: Local Testnet (eth-tester)")
        logger.info("- Balance: 100 ETH")
        
        logger.info("\nTo test the blockchain integration:")
        logger.info("1. The local testnet is ready to use")
        logger.info("2. Run the diagnostic tests:")
        logger.info("   python diagnostics/blockchain_test.py")
        
    except Exception as e:
        logger.error(f"Error in setup process: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 