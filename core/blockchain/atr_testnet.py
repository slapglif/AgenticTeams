"""ATR Framework testnet interaction script."""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple
from web3 import Web3, EthereumTesterProvider
from eth_tester import EthereumTester, PyEVMBackend
import solcx
from eth_typing import HexStr
from eth_utils import to_bytes, remove_0x_prefix, to_hex
from solcx import compile_source, install_solc, set_solc_version
from loguru import logger
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler

# Configure rich
console = Console()
install(show_locals=True)

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    RichHandler(
        console=console,
        show_path=False,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True
    ),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/atr_testnet_{time}.log",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

class ATRTestnet:
    """Class for managing ATR Framework contract interactions on testnet."""
    
    def __init__(self):
        """Initialize the ATR Framework testnet interaction."""
        # Initialize paths and configuration
        self.workspace_root = Path(__file__).parent.parent
        
        # Initialize contract instances and addresses
        self.agent_registry = None
        self.agent_registry_address = None
        self.work_unit_registry = None
        self.work_unit_registry_address = None
        self.responsibility_nft = None
        self.responsibility_nft_address = None

        # Set up environment
        self._setup_solidity()
        
        # Initialize Web3 with eth-tester
        self.eth_tester = EthereumTester(PyEVMBackend())
        self.w3 = Web3(EthereumTesterProvider(self.eth_tester))
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        
        # Deploy contracts
        self.deploy_contracts()
    
    def _setup_solidity(self):
        """Set up Solidity compiler."""
        try:
            solcx.install_solc('0.8.24')
            solcx.set_solc_version('0.8.24')
            logger.info("Solidity compiler setup complete")
        except Exception as e:
            logger.error(f"Failed to set up Solidity compiler: {str(e)}")
            raise
    
    def compile_contract(self, contract_name: str) -> Tuple[dict, str]:
        """Compile a Solidity contract and return its ABI and bytecode."""
        contract_dir = self.workspace_root / "blockchain" / "contracts"
        contract_file = contract_dir / f"{contract_name}.sol"

        if not contract_file.exists():
            raise FileNotFoundError(f"Contract file {contract_file} not found")

        with open(contract_file, 'r', encoding='utf-8') as f:
            source = f.read()

        input_data = {
            "language": "Solidity",
            "sources": {
                os.path.basename(contract_file): {"content": source}
            },
            "settings": {
                "optimizer": {
                    "enabled": True,
                    "runs": 200
                },
                "viaIR": True,
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode"]
                    }
                }
            }
        }

        try:
            compiled_sol = solcx.compile_standard(
                input_data,
                base_path=str(contract_dir),
                allow_paths=[str(contract_dir)]
            )

            contract_interface = compiled_sol['contracts'][os.path.basename(contract_file)][contract_name]
            abi = contract_interface['abi']
            bytecode = contract_interface['evm']['bytecode']['object']
            return abi, bytecode

        except Exception as e:
            logger.error(f"Failed to compile contract {contract_name}: {str(e)}")
            raise
    
    def deploy_contract(self, contract_name: str, constructor_args: list = None) -> Tuple[Any, str]:
        """Deploy a contract and return its instance and address."""
        try:
            abi, bytecode = self.compile_contract(contract_name)
            contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Deploy the contract
            if constructor_args is None:
                constructor_args = []
            tx_hash = contract.constructor(*constructor_args).transact({'from': self.w3.eth.default_account})
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Create contract instance at deployed address
            contract_instance = self.w3.eth.contract(address=tx_receipt.contractAddress, abi=abi)
            return contract_instance, tx_receipt.contractAddress

        except Exception as e:
            logger.error(f"Failed to deploy contract {contract_name}: {str(e)}")
            raise
    
    def deploy_contracts(self):
        """Deploy ATR Framework contracts."""
        try:
            # Deploy AgentRegistry
            self.agent_registry, self.agent_registry_address = self.deploy_contract(
                "AgentRegistry",
                constructor_args=[]
            )
            logger.info(f"Deployed AgentRegistry to {self.agent_registry_address}")

            # Deploy WorkUnitRegistry
            self.work_unit_registry, self.work_unit_registry_address = self.deploy_contract(
                "WorkUnitRegistry",
                constructor_args=[self.agent_registry_address]
            )
            logger.info(f"Deployed WorkUnitRegistry to {self.work_unit_registry_address}")

            # Deploy ResponsibilityNFT
            self.responsibility_nft, self.responsibility_nft_address = self.deploy_contract(
                "ResponsibilityNFT",
                constructor_args=["ATR Responsibility", "ATRR"]
            )
            logger.info(f"Deployed ResponsibilityNFT to {self.responsibility_nft_address}")

            logger.info("ATR Framework contracts deployed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy ATR Framework: {str(e)}")
            raise
    
    def _generate_agent_uuid(self) -> bytes:
        """Generate a random UUID and convert it to bytes32 format."""
        # Generate a random UUID
        agent_uuid = uuid.uuid4()
        # Convert UUID to bytes and pad to 32 bytes
        return agent_uuid.bytes.rjust(32, b'\0')

    def _uuid_to_hex(self, uuid_obj):
        """Convert a UUID to bytes32 format."""
        if isinstance(uuid_obj, bytes):
            # Pad bytes to 32 bytes
            return uuid_obj.rjust(32, b'\0')
        elif isinstance(uuid_obj, uuid.UUID):
            # Convert UUID to bytes and pad to 32 bytes
            return uuid_obj.bytes.rjust(32, b'\0')
        elif isinstance(uuid_obj, str):
            # Convert string UUID to bytes and pad to 32 bytes
            return uuid.UUID(uuid_obj).bytes.rjust(32, b'\0')
        else:
            raise ValueError(f"Unsupported type for UUID conversion: {type(uuid_obj)}")

    def _hex_to_uuid(self, hex_str):
        """Convert a hex string to a UUID."""
        if isinstance(hex_str, bytes):
            # Convert bytes to hex string
            hex_str = '0x' + hex_str.hex()
        
        clean_hex = remove_0x_prefix(hex_str)[-32:]  # Take last 32 chars (16 bytes)
        return uuid.UUID(clean_hex)

    def register_agent_onchain(self, agent_type: str, capabilities: list) -> uuid.UUID:
        """
        Register an agent on the blockchain.
        
        Args:
            agent_type: Type/category of the agent
            capabilities: List of agent capabilities
            
        Returns:
            uuid.UUID: The generated agent UUID
        """
        try:
            # Generate UUID and convert to bytes32
            agent_uuid_bytes = self._generate_agent_uuid()
            agent_uuid_hex = self._uuid_to_hex(agent_uuid_bytes)
            
            tx_hash = self.agent_registry.functions.registerAgent(
                agent_uuid_hex,
                agent_type,
                capabilities
            ).transact({'from': self.w3.eth.default_account})
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                # Get the agent ID from the event logs
                event = self.agent_registry.events.AgentRegistered().process_receipt(receipt)[0]
                agent_id_hex = event.args.agentId
                agent_uuid = self._hex_to_uuid(agent_id_hex)
                logger.success(f"Agent registered successfully with UUID {agent_uuid}")
                return agent_uuid
            else:
                raise Exception("Agent registration transaction failed")
            
        except Exception as e:
            logger.error(f"Failed to register agent: {str(e)}")
            raise
    
    def get_agent_details_onchain(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get agent details from the blockchain.
        
        Args:
            agent_id: UUID of the agent to query
            
        Returns:
            Dict containing agent details
        """
        try:
            # Convert UUID to bytes32 hex
            agent_id_hex = self._uuid_to_hex(agent_id)
            agent_details = self.agent_registry.functions.getAgent(agent_id_hex).call()
            
            # Convert the returned bytes32 back to UUID
            returned_uuid = self._hex_to_uuid(agent_details[0])
            
            return {
                'agent_id': str(returned_uuid),
                'agent_type': agent_details[1],
                'capabilities': agent_details[2],
                'is_active': agent_details[3],
                'registered_at': agent_details[4],
                'owner': agent_details[5]
            }
        except Exception as e:
            logger.error(f"Failed to get agent details for {agent_id}: {str(e)}")
            raise
    
    def define_work_unit_onchain(
        self,
        name: str,
        description: str,
        version: str,
        inputs: str,
        outputs: str,
        requirements: str,
        temporal_constraints: str,
        data_access_controls: str,
        payment_token: str,
        payment_amount: int
    ) -> int:
        """
        Define a work unit on the blockchain.
        
        Args:
            name: Name of the work unit
            description: Description of the work unit
            version: Version of the work unit
            inputs: JSON string of input specifications
            outputs: JSON string of output specifications
            requirements: JSON string of agent requirements
            temporal_constraints: JSON string of temporal constraints
            data_access_controls: JSON string of data access controls
            payment_token: Address of token used for payment
            payment_amount: Amount to be paid for completion
            
        Returns:
            ID of the created work unit
        """
        try:
            tx_hash = self.work_unit_registry.functions.createWorkUnit(
                name,
                description,
                version,
                inputs,
                outputs,
                requirements,
                temporal_constraints,
                data_access_controls,
                payment_token,
                payment_amount
            ).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Get work unit ID from event logs
            event = self.work_unit_registry.events.WorkUnitCreated().process_receipt(tx_receipt)[0]
            work_unit_id = event['args']['id']
            
            logger.success(f"Work unit {work_unit_id} created successfully")
            return work_unit_id
        except Exception as e:
            logger.error(f"Failed to create work unit: {str(e)}")
            raise
    
    def get_work_unit_onchain(self, work_unit_id: int) -> Dict[str, Any]:
        """
        Get work unit details from the blockchain.
        
        Args:
            work_unit_id: ID of the work unit to query
            
        Returns:
            Dict containing work unit details
        """
        try:
            work_unit = self.work_unit_registry.functions.getWorkUnit(work_unit_id).call()
            return {
                'id': work_unit[0],
                'name': work_unit[1],
                'description': work_unit[2],
                'version': work_unit[3],
                'inputs': work_unit[4],
                'outputs': work_unit[5],
                'requirements': work_unit[6],
                'temporal_constraints': work_unit[7],
                'data_access_controls': work_unit[8],
                'payment_token': work_unit[9],
                'payment_amount': work_unit[10],
                'owner': work_unit[11],
                'created_at': work_unit[12],
                'is_active': work_unit[13],
                'assigned_agent_id': work_unit[14].hex() if work_unit[14] else None,
                'status': work_unit[15]
            }
        except Exception as e:
            logger.error(f"Failed to get work unit {work_unit_id}: {str(e)}")
            raise
    
    def assign_agent_to_work_unit_onchain(
        self,
        work_unit_id: int,
        agent_id: uuid.UUID
    ) -> bool:
        """
        Assign an agent to a work unit.
        
        Args:
            work_unit_id: ID of the work unit
            agent_id: UUID of the agent to assign
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert UUID to bytes32
            agent_id_hex = self._uuid_to_hex(agent_id)
            
            # Assign agent to work unit
            tx_hash = self.work_unit_registry.functions.assignAgent(
                work_unit_id,
                agent_id_hex
            ).transact({'from': self.w3.eth.default_account})
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                logger.info(f"Agent {agent_id} assigned to work unit {work_unit_id}")
                return True
            else:
                raise Exception("Agent assignment transaction failed")
            
        except Exception as e:
            logger.error(f"Failed to assign agent to work unit: {str(e)}")
            raise

    def mint_responsibility_nft(
        self,
        agent_id: uuid.UUID,
        work_unit_id: int,
        action_type: str,
        metadata: Dict[str, Any]
    ) -> int:
        """
        Mint a new responsibility NFT.
        
        Args:
            agent_id: UUID of the agent
            work_unit_id: ID of the work unit
            action_type: Type of action (e.g., "assignment", "completion")
            metadata: Additional metadata for the NFT
            
        Returns:
            int: Token ID of the minted NFT
        """
        try:
            # Convert UUID to bytes32
            agent_id_hex = self._uuid_to_hex(agent_id)
            
            # Convert metadata to URI format (in real implementation, this would be stored on IPFS)
            metadata_uri = f"ipfs://placeholder/{work_unit_id}/{action_type}"
            
            # Mint the NFT
            tx_hash = self.responsibility_nft.functions.mintResponsibility(
                agent_id_hex,
                work_unit_id,
                action_type,
                metadata_uri
            ).transact({'from': self.w3.eth.default_account})
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                # Get the token ID from the event logs
                event = self.responsibility_nft.events.ResponsibilityMinted().process_receipt(receipt)[0]
                token_id = event.args.tokenId
                logger.success(f"Minted responsibility NFT with token ID {token_id}")
                return token_id
            else:
                raise Exception("NFT minting transaction failed")
            
        except Exception as e:
            logger.error(f"Failed to mint responsibility NFT: {str(e)}")
            raise

    def update_work_unit_status_onchain(
        self,
        work_unit_id: int,
        new_status: str,
        status_metadata: Dict[str, Any]
    ) -> bool:
        """
        Update the status of a work unit.
        
        Args:
            work_unit_id: ID of the work unit
            new_status: New status to set (one of: CREATED, ASSIGNED, IN_PROGRESS, COMPLETED, FAILED, VERIFIED)
            status_metadata: Additional metadata about the status change
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert status string to uint8
            status_map = {
                'CREATED': 0,
                'ASSIGNED': 1,
                'IN_PROGRESS': 2,
                'COMPLETED': 3,
                'FAILED': 4,
                'VERIFIED': 5
            }
            status_uint8 = status_map[new_status.upper()]
            
            # Convert metadata to JSON string
            metadata_str = json.dumps(status_metadata)
            
            # Update work unit status
            tx_hash = self.work_unit_registry.functions.updateStatus(
                work_unit_id,
                status_uint8,
                metadata_str
            ).transact({'from': self.w3.eth.default_account})
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                logger.info(f"Updated work unit {work_unit_id} status to {new_status}")
                return True
            else:
                raise Exception("Work unit status update transaction failed")
            
        except Exception as e:
            logger.error(f"Failed to update work unit status: {str(e)}")
            raise

    async def get_task_onchain(self, task_id: str) -> Dict[str, Any]:
        """Get task data from the blockchain.
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Task data including status, metrics, and results
        """
        try:
            # For testing, return mock data
            return {
                "task_id": task_id,
                "status": "Completed",
                "metrics": {
                    "completionScore": 0.85,
                    "qualityScore": 0.78,
                    "timeEfficiency": 0.92
                },
                "plan": {
                    "tasks": [
                        {
                            "id": "op_1",
                            "description": "Initial research",
                            "tool_name": "search",
                            "dependencies": [],
                            "args": {"query": "test query"}
                        }
                    ]
                },
                "results": {
                    "final_result": "Test result",
                    "execution_results": []
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting task from blockchain: {e}")
            raise

    async def update_task_result_onchain(self, task_id: str, result: Any, metrics: Dict[str, Any]) -> None:
        """Update task result and metrics on the blockchain.
        
        Args:
            task_id: ID of the task to update
            result: Task execution result
            metrics: Task execution metrics
        """
        try:
            # For testing, just log the update
            logger.info(f"Updating task {task_id} on blockchain")
            logger.debug(f"Result: {result}")
            logger.debug(f"Metrics: {metrics}")
            
            # In a real implementation, this would update the task contract
            # await self.work_unit_registry.update_task_result(
            #     task_id=task_id,
            #     result=result,
            #     metrics=metrics
            # )
            
        except Exception as e:
            logger.error(f"Error updating task result on blockchain: {e}")
            raise

def main():
    """Main function to test ATR Framework contracts."""
    try:
        console.rule("[bold blue]Deploying ATR Framework Contracts[/bold blue]")
        atr = ATRTestnet()

        # Register a test agent
        console.rule("[bold green]Registering Test Agent[/bold green]")
        agent_id = uuid.uuid4()
        agent_id_bytes = atr._uuid_to_hex(agent_id)
        tx_hash = atr.agent_registry.functions.registerAgent(
            agent_id_bytes,
            "TestAgent",
            ["test_capability"]
        ).transact()
        atr.w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.success(f"Agent registered with ID: {agent_id}")

        # Get and log agent details
        agent_details = atr.get_agent_details_onchain(agent_id)
        console.print("[bold cyan]Agent Details:[/bold cyan]")
        console.print_json(data=agent_details)

        # Create a test work unit
        console.rule("[bold yellow]Creating Test Work Unit[/bold yellow]")
        work_unit_id = atr.define_work_unit_onchain(
            name="Test Work Unit",
            description="Test Description",
            version="1.0.0",
            inputs="test_inputs",
            outputs="test_outputs",
            requirements="test_requirements",
            temporal_constraints="test_temporal_constraints",
            data_access_controls="test_data_access_controls",
            payment_token="ETH",
            payment_amount=100
        )

        logger.success(f"Work unit {work_unit_id} created successfully")

        # Get and log work unit details
        work_unit = atr.get_work_unit_onchain(work_unit_id)
        console.print("[bold magenta]Work Unit Details:[/bold magenta]")
        console.print_json(data=work_unit)

    except Exception as e:
        logger.exception("Error in main")
        raise

if __name__ == "__main__":
    main() 