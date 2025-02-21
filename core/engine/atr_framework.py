"""ATR Framework core module for managing agentic work units."""

import json
import logging
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from core.memory.memory_manager import MemoryManager
from core.engine.agent_manager import AgentManager
from core.shared.settings import build_llm
from core.shared.data_processor import IntelligentDataProcessor
from core.blockchain.blockchain_manager import BlockchainManager

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status of an agentic work unit."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class ResponsibilityNFT:
    """Represents a responsibility NFT for tracking agent contributions."""
    token_id: str
    agent_id: str
    work_unit_id: str
    action_type: str
    timestamp: str
    metadata: Dict[str, Any]

class ATRFramework:
    """Core class for the Agentic Task Registry Framework."""
    
    def __init__(self, network_url: Optional[str] = None, contract_address: Optional[str] = None):
        """Initialize the ATR Framework."""
        self.logger = logging.getLogger(__name__)
        self.memory_manager = MemoryManager()
        self.agent_manager = AgentManager()
        self.data_processor = IntelligentDataProcessor()
        
        # Initialize LLM for task processing
        self.llm = build_llm(output_mode='json', temperature=0.2)
        
        # Initialize storage for work units and responsibilities
        self.work_units: Dict[str, Dict[str, Any]] = {}
        self.responsibility_nfts: Dict[str, ResponsibilityNFT] = {}
        
        # Initialize blockchain manager if network URL provided
        self.blockchain_manager = None
        if network_url:
            self.blockchain_manager = BlockchainManager(network_url, contract_address)
            self.logger.info("Initialized blockchain integration")
        
    async def register_work_unit(self, specification: Dict[str, Any]) -> str:
        """Register a new agentic work unit.
        
        Args:
            specification: Work unit specification following ATR schema
            
        Returns:
            str: Work unit ID
        """
        try:
            # Validate specification
            await self._validate_work_unit_spec(specification)
            
            # Generate work unit ID
            work_unit_id = f"wu_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(self.work_units)}"
            
            # Create work unit record
            work_unit = {
                "id": work_unit_id,
                "specification": specification,
                "status": TaskStatus.PENDING.value,
                "assigned_agents": [],
                "temporal_constraints": specification.get("temporal_constraints", {}),
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat()
            }
            
            # Store in memory
            self.work_units[work_unit_id] = work_unit
            await self.memory_manager.store_work_unit(work_unit)
            
            self.logger.info(f"Registered work unit {work_unit_id}")
            return work_unit_id
            
        except Exception as e:
            self.logger.error(f"Error registering work unit: {e}")
            raise
            
    async def assign_agent(self, work_unit_id: str, agent_id: str) -> bool:
        """Assign an agent to a work unit.
        
        Args:
            work_unit_id: ID of the work unit
            agent_id: ID of the agent to assign
            
        Returns:
            bool: True if assignment successful
        """
        try:
            if work_unit_id not in self.work_units:
                raise ValueError(f"Work unit {work_unit_id} not found")
                
            work_unit = self.work_units[work_unit_id]
            
            # Verify agent capabilities match requirements
            await self._verify_agent_capabilities(agent_id, work_unit["specification"])
            
            # Check temporal constraints
            if not await self._check_temporal_constraints(agent_id, work_unit):
                raise ValueError(f"Agent {agent_id} cannot meet temporal constraints")
                
            # Update work unit
            work_unit["assigned_agents"].append(agent_id)
            work_unit["updated_at"] = datetime.now(UTC).isoformat()
            
            # Update memory
            await self.memory_manager.update_work_unit(work_unit)
            
            # Initialize agent context
            await self.memory_manager.initialize_agent_memory(
                agent_id=agent_id,
                agent_type="work_unit_agent",
                capabilities=work_unit["specification"].get("required_capabilities", [])
            )
            
            # Mint responsibility NFT for assignment if blockchain enabled
            if self.blockchain_manager:
                token_id = await self.blockchain_manager.mint_responsibility_nft(
                    agent_id=agent_id,
                    work_unit_id=work_unit_id,
                    action_type="assignment",
                    metadata={
                        "assignment_time": datetime.now(UTC).isoformat(),
                        "requirements": work_unit["specification"].get("requirements", {})
                    }
                )
                
                # Store NFT data in memory
                nft = ResponsibilityNFT(
                    token_id=token_id,
                    agent_id=agent_id,
                    work_unit_id=work_unit_id,
                    action_type="assignment",
                    timestamp=datetime.now(UTC).isoformat(),
                    metadata={"assignment_time": datetime.now(UTC).isoformat()}
                )
                self.responsibility_nfts[token_id] = nft
                await self.memory_manager.store_responsibility_nft(nft.__dict__)
            
            self.logger.info(f"Assigned agent {agent_id} to work unit {work_unit_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error assigning agent: {e}")
            return False
            
    async def mint_responsibility_nft(
        self,
        agent_id: str,
        work_unit_id: str,
        action_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Mint a new responsibility NFT for an agent's contribution.
        
        Args:
            agent_id: ID of the agent
            work_unit_id: ID of the work unit
            action_type: Type of action performed
            metadata: Additional metadata about the action
            
        Returns:
            str: Token ID of the minted NFT
        """
        try:
            if not self.blockchain_manager:
                raise ValueError("Blockchain integration not initialized")
                
            # Mint NFT
            token_id = await self.blockchain_manager.mint_responsibility_nft(
                agent_id=agent_id,
                work_unit_id=work_unit_id,
                action_type=action_type,
                metadata=metadata or {}
            )
            
            # Create NFT record
            nft = ResponsibilityNFT(
                token_id=token_id,
                agent_id=agent_id,
                work_unit_id=work_unit_id,
                action_type=action_type,
                timestamp=datetime.now(UTC).isoformat(),
                metadata=metadata or {}
            )
            
            # Store in memory
            self.responsibility_nfts[token_id] = nft
            await self.memory_manager.store_responsibility_nft(nft.__dict__)
            
            self.logger.info(f"Minted responsibility NFT {token_id}")
            return token_id
            
        except Exception as e:
            self.logger.error(f"Error minting responsibility NFT: {e}")
            raise
            
    async def update_work_unit_status(
        self,
        work_unit_id: str,
        status: TaskStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update the status of a work unit.
        
        Args:
            work_unit_id: ID of the work unit
            status: New status
            metadata: Additional metadata about the status change
            
        Returns:
            bool: True if update successful
        """
        try:
            if work_unit_id not in self.work_units:
                raise ValueError(f"Work unit {work_unit_id} not found")
                
            work_unit = self.work_units[work_unit_id]
            
            # Update status
            work_unit["status"] = status.value
            work_unit["updated_at"] = datetime.now(UTC).isoformat()
            if metadata:
                work_unit["status_metadata"] = metadata
                
            # Update memory
            await self.memory_manager.update_work_unit(work_unit)
            
            # Mint NFT for status change if blockchain enabled
            if self.blockchain_manager and work_unit["assigned_agents"]:
                for agent_id in work_unit["assigned_agents"]:
                    await self.mint_responsibility_nft(
                        agent_id=agent_id,
                        work_unit_id=work_unit_id,
                        action_type=f"status_change_{status.value}",
                        metadata={
                            "previous_status": work_unit.get("status"),
                            "new_status": status.value,
                            "change_time": datetime.now(UTC).isoformat(),
                            **(metadata or {})
                        }
                    )
            
            self.logger.info(f"Updated work unit {work_unit_id} status to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating work unit status: {e}")
            return False
            
    async def verify_temporal_constraints(self, work_unit_id: str) -> bool:
        """Verify that a work unit's temporal constraints are being met.
        
        Args:
            work_unit_id: ID of the work unit
            
        Returns:
            bool: True if constraints are met
        """
        try:
            if work_unit_id not in self.work_units:
                raise ValueError(f"Work unit {work_unit_id} not found")
                
            work_unit = self.work_units[work_unit_id]
            constraints = work_unit.get("temporal_constraints", {})
            
            if not constraints:
                return True
                
            now = datetime.now(UTC)
            
            # Check start time
            if "start_time" in constraints:
                start_time = datetime.fromisoformat(constraints["start_time"])
                if now < start_time:
                    return False
                    
            # Check deadline
            if "deadline" in constraints:
                deadline = datetime.fromisoformat(constraints["deadline"])
                if now > deadline:
                    return False
                    
            # Check duration constraints
            if "max_duration" in constraints:
                start_time = datetime.fromisoformat(work_unit["created_at"])
                max_duration = float(constraints["max_duration"])  # in hours
                elapsed = (now - start_time).total_seconds() / 3600
                if elapsed > max_duration:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying temporal constraints: {e}")
            return False
            
    async def _validate_work_unit_spec(self, specification: Dict[str, Any]) -> bool:
        """Validate a work unit specification."""
        try:
            # TODO: Implement validation against ATR schema
            # For now, just check required fields
            required_fields = [
                "name",
                "description",
                "inputs",
                "outputs",
                "requirements"
            ]
            
            for field in required_fields:
                if field not in specification:
                    raise ValueError(f"Missing required field: {field}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating work unit specification: {e}")
            raise
            
    async def _verify_agent_capabilities(
        self,
        agent_id: str,
        specification: Dict[str, Any]
    ) -> bool:
        """Verify an agent has the required capabilities for a work unit."""
        try:
            # Get agent capabilities
            agent_context = await self.memory_manager.get_agent_context(agent_id)
            if not agent_context:
                raise ValueError(f"Agent {agent_id} not found")
                
            agent_capabilities = agent_context.get("working_memory", {}).get("capabilities", [])
            
            # Check against required capabilities
            required_capabilities = specification.get("requirements", {}).get("capabilities", [])
            
            for capability in required_capabilities:
                if capability not in agent_capabilities:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying agent capabilities: {e}")
            return False
            
    async def _check_temporal_constraints(
        self,
        agent_id: str,
        work_unit: Dict[str, Any]
    ) -> bool:
        """Check if an agent can meet a work unit's temporal constraints."""
        try:
            # Get agent's current commitments
            agent_context = await self.memory_manager.get_agent_context(agent_id)
            if not agent_context:
                raise ValueError(f"Agent {agent_id} not found")
                
            # TODO: Implement more sophisticated temporal constraint checking
            # For now, just check if agent is already assigned to max work units
            current_assignments = len([
                wu for wu in self.work_units.values()
                if agent_id in wu["assigned_agents"] and
                wu["status"] in [TaskStatus.PENDING.value, TaskStatus.IN_PROGRESS.value]
            ])
            
            MAX_CONCURRENT_ASSIGNMENTS = 5  # TODO: Make configurable
            
            return current_assignments < MAX_CONCURRENT_ASSIGNMENTS
            
        except Exception as e:
            self.logger.error(f"Error checking temporal constraints: {e}")
            return False 