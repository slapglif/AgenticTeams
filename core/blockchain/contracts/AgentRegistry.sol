// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AgentRegistry
 * @dev Contract for registering and managing AI agents in the ATR Framework
 */
contract AgentRegistry {
    // Structs
    struct Agent {
        bytes32 agentId;       // UUID for the agent stored as bytes32
        string agentType;      // Type/category of the agent
        string[] capabilities; // List of agent capabilities
        bool isActive;         // Whether the agent is currently active
        uint256 registeredAt;  // Timestamp when the agent was registered
        address owner;          // Address that registered the agent
    }

    // State variables
    mapping(bytes32 => Agent) private agents;
    mapping(bytes32 => bool) private usedIds;
    
    // Array to keep track of all agent IDs
    bytes32[] private agentIds;

    // Events
    event AgentRegistered(bytes32 agentId, string agentType, string[] capabilities, address owner);
    event AgentDeactivated(bytes32 agentId);
    event AgentReactivated(bytes32 agentId);
    event CapabilitiesUpdated(bytes32 agentId, string[] capabilities);

    // Modifiers
    modifier onlyAgentOwner(bytes32 agentId) {
        require(agents[agentId].owner == msg.sender, "Only agent owner can perform this action");
        _;
    }

    modifier agentExists(bytes32 agentId) {
        require(usedIds[agentId], "Agent does not exist");
        _;
    }

    /**
     * @dev Register a new agent
     * @param agentId UUID for the agent as bytes32
     * @param agentType Type/category of the agent
     * @param capabilities List of agent capabilities
     */
    function registerAgent(
        bytes32 agentId,
        string memory agentType,
        string[] memory capabilities
    ) public {
        require(bytes(agentType).length > 0, "Agent type cannot be empty");
        require(capabilities.length > 0, "Must specify at least one capability");
        require(!usedIds[agentId], "Agent ID already exists");
        require(agentId != bytes32(0), "Invalid agent ID");

        // Create and store new agent
        agents[agentId] = Agent({
            agentId: agentId,
            agentType: agentType,
            capabilities: capabilities,
            isActive: true,
            registeredAt: block.timestamp,
            owner: msg.sender
        });

        usedIds[agentId] = true;
        agentIds.push(agentId);
        emit AgentRegistered(agentId, agentType, capabilities, msg.sender);
    }

    /**
     * @dev Get agent details
     * @param agentId The UUID of the agent
     * @return Agent details (id, type, capabilities, isActive, registeredAt, owner)
     */
    function getAgent(bytes32 agentId) 
        public 
        view 
        agentExists(agentId) 
        returns (
            bytes32,
            string memory,
            string[] memory,
            bool,
            uint256,
            address
        ) 
    {
        Agent storage agent = agents[agentId];
        return (
            agent.agentId,
            agent.agentType,
            agent.capabilities,
            agent.isActive,
            agent.registeredAt,
            agent.owner
        );
    }

    /**
     * @dev Deactivate an agent
     * @param agentId The UUID of the agent to deactivate
     */
    function deactivateAgent(bytes32 agentId) 
        public 
        onlyAgentOwner(agentId) 
        agentExists(agentId) 
    {
        require(agents[agentId].isActive, "Agent is already inactive");
        agents[agentId].isActive = false;
        emit AgentDeactivated(agentId);
    }

    /**
     * @dev Reactivate an agent
     * @param agentId The UUID of the agent to reactivate
     */
    function reactivateAgent(bytes32 agentId) 
        public 
        onlyAgentOwner(agentId) 
        agentExists(agentId) 
    {
        require(!agents[agentId].isActive, "Agent is already active");
        agents[agentId].isActive = true;
        emit AgentReactivated(agentId);
    }

    /**
     * @dev Update agent capabilities
     * @param agentId The UUID of the agent
     * @param capabilities New list of capabilities
     */
    function updateCapabilities(bytes32 agentId, string[] memory capabilities) 
        public 
        onlyAgentOwner(agentId) 
        agentExists(agentId) 
    {
        require(capabilities.length > 0, "Must specify at least one capability");
        agents[agentId].capabilities = capabilities;
        emit CapabilitiesUpdated(agentId, capabilities);
    }

    /**
     * @dev Get total number of registered agents
     * @return Total number of agents
     */
    function getTotalAgents() public view returns (uint256) {
        return agentIds.length;
    }

    /**
     * @dev Check if agent has a specific capability
     * @param agentId The UUID of the agent
     * @param capability The capability to check for
     * @return bool indicating if agent has the capability
     */
    function hasCapability(bytes32 agentId, string memory capability) 
        public 
        view 
        agentExists(agentId) 
        returns (bool) 
    {
        string[] memory agentCapabilities = agents[agentId].capabilities;
        for (uint i = 0; i < agentCapabilities.length; i++) {
            if (keccak256(abi.encodePacked(agentCapabilities[i])) == keccak256(abi.encodePacked(capability))) {
                return true;
            }
        }
        return false;
    }
} 