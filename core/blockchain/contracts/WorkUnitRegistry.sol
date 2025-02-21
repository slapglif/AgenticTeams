// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./AgentRegistry.sol";

/**
 * @title WorkUnitRegistry
 * @dev Contract for registering and managing work units in the ATR Framework
 */
contract WorkUnitRegistry {
    struct WorkUnit {
        uint256 id;                  // Unique identifier for the work unit
        string name;                 // Name of the work unit
        string description;          // Description of the work unit
        string version;              // Version of the work unit
        string inputs;               // JSON string of input specifications
        string outputs;              // JSON string of output specifications
        string requirements;         // JSON string of agent requirements
        string temporalConstraints;  // JSON string of temporal constraints
        string dataAccessControls;   // JSON string of data access controls
        string paymentToken;         // Token used for payment (address(0) for native token)
        uint256 paymentAmount;        // Amount to be paid for completion
        address owner;               // Address that created the work unit
        uint256 createdAt;          // Timestamp when the work unit was created
        bool isActive;               // Whether the work unit is currently active
        bytes32 assignedAgentId;      // ID of the assigned agent (empty if unassigned)
        WorkUnitStatus status;       // Current status of the work unit
    }

    enum WorkUnitStatus {
        UNDEFINED,
        CREATED,
        ASSIGNED,
        IN_PROGRESS,
        COMPLETED,
        VERIFIED,
        FAILED,
        CANCELLED,
        DISPUTED
    }

    // Reference to the AgentRegistry contract
    AgentRegistry public agentRegistry;

    // Counter for work unit IDs
    uint256 private workUnitCounter;

    // Mapping from work unit ID to WorkUnit struct
    mapping(uint256 => WorkUnit) public workUnits;

    // Events
    event WorkUnitCreated(uint256 indexed id, string name, address owner);
    event WorkUnitAssigned(uint256 indexed id, bytes32 indexed agentId);
    event WorkUnitStatusUpdated(
        uint256 indexed id,
        WorkUnitStatus oldStatus,
        WorkUnitStatus newStatus,
        bytes32 indexed agentId,
        uint256 timestamp
    );
    event WorkUnitDeactivated(uint256 indexed id);
    event WorkUnitReactivated(uint256 indexed id);

    // Modifiers
    modifier onlyWorkUnitOwner(uint256 workUnitId) {
        require(workUnits[workUnitId].owner == msg.sender, "Not the work unit owner");
        _;
    }

    modifier workUnitExists(uint256 workUnitId) {
        require(workUnitId > 0 && workUnitId <= workUnitCounter, "Work unit does not exist");
        _;
    }

    /**
     * @dev Constructor
     * @param _agentRegistry Address of the AgentRegistry contract
     */
    constructor(address _agentRegistry) {
        require(_agentRegistry != address(0), "Invalid agent registry address");
        agentRegistry = AgentRegistry(_agentRegistry);
        workUnitCounter = 0;
    }

    /**
     * @dev Create a new work unit
     * @param name Name of the work unit
     * @param description Description of the work unit
     * @param version Version of the work unit
     * @param inputs JSON string of input specifications
     * @param outputs JSON string of output specifications
     * @param requirements JSON string of agent requirements
     * @param temporalConstraints JSON string of temporal constraints
     * @param dataAccessControls JSON string of data access controls
     * @param paymentToken Token used for payment (address(0) for native token)
     * @param paymentAmount Amount to be paid for completion
     * @return id of the created work unit
     */
    function createWorkUnit(
        string memory name,
        string memory description,
        string memory version,
        string memory inputs,
        string memory outputs,
        string memory requirements,
        string memory temporalConstraints,
        string memory dataAccessControls,
        string memory paymentToken,
        uint256 paymentAmount
    ) public returns (uint256) {
        require(bytes(name).length > 0, "Name cannot be empty");
        require(bytes(description).length > 0, "Description cannot be empty");
        require(bytes(version).length > 0, "Version cannot be empty");
        require(bytes(requirements).length > 0, "Requirements cannot be empty");
        require(bytes(temporalConstraints).length > 0, "Temporal constraints cannot be empty");
        require(paymentAmount > 0, "Payment amount must be greater than 0");

        workUnitCounter++;
        uint256 workUnitId = workUnitCounter;

        WorkUnit storage workUnit = workUnits[workUnitId];
        workUnit.id = workUnitId;
        workUnit.name = name;
        workUnit.description = description;
        workUnit.version = version;
        workUnit.inputs = inputs;
        workUnit.outputs = outputs;
        workUnit.requirements = requirements;
        workUnit.temporalConstraints = temporalConstraints;
        workUnit.dataAccessControls = dataAccessControls;
        workUnit.paymentToken = paymentToken;
        workUnit.paymentAmount = paymentAmount;
        workUnit.owner = msg.sender;
        workUnit.createdAt = block.timestamp;
        workUnit.isActive = true;
        workUnit.assignedAgentId = bytes32(0);  // Initialize with zero bytes
        workUnit.status = WorkUnitStatus.CREATED;

        emit WorkUnitCreated(workUnitId, name, msg.sender);
        return workUnitId;
    }

    /**
     * @dev Get work unit details
     * @param workUnitId The ID of the work unit
     * @return WorkUnit details
     */
    function getWorkUnit(uint256 workUnitId) public view workUnitExists(workUnitId) returns (WorkUnit memory) {
        return workUnits[workUnitId];
    }

    /**
     * @dev Helper function to parse temporal constraints JSON
     * @param jsonStr The temporal constraints JSON string
     * @return startTime The start time timestamp
     * @return deadline The deadline timestamp
     */
    function parseTemporalConstraints(string memory jsonStr) internal pure returns (uint256 startTime, uint256 deadline) {
        // Note: In production, use a proper JSON parser library
        // For this testnet implementation, we use a simple string parsing approach
        bytes memory strBytes = bytes(jsonStr);
        require(strBytes.length > 0, "Empty temporal constraints");
        
        // Extract values using string operations
        // Format expected: {"start_time": 123, "deadline": 456}
        uint256 start = _findValueAfter(strBytes, "start_time");
        uint256 end = _findValueAfter(strBytes, "deadline");
        
        return (start, end);
    }

    /**
     * @dev Parse agent requirements from JSON string
     * @param requirementsJson JSON string containing agent requirements
     * @return requiredType The required agent type
     */
    function parseAgentRequirements(string memory requirementsJson) internal pure returns (string memory) {
        bytes memory requirements = bytes(requirementsJson);
        require(requirements.length > 0, "Empty requirements");
        
        // Find the agent_type field with escaped quotes
        bytes memory searchStr = bytes('agent_type');
        uint startPos = 0;
        bool found = false;
        
        // Find the start position of agent_type field
        for (uint i = 0; i < requirements.length - searchStr.length; i++) {
            bool isMatch = true;
            for (uint j = 0; j < searchStr.length; j++) {
                if (requirements[i + j] != searchStr[j]) {
                    isMatch = false;
                    break;
                }
            }
            if (isMatch) {
                startPos = i + searchStr.length;
                found = true;
                break;
            }
        }
        require(found, "agent_type field not found");
        
        // Find the colon and opening quote, skipping any escaped quotes
        found = false;
        for (uint i = startPos; i < requirements.length - 1; i++) {
            if (requirements[i] == ':') {
                // Skip whitespace
                uint j = i + 1;
                while (j < requirements.length && (requirements[j] == ' ' || requirements[j] == '"')) {
                    j++;
                }
                startPos = j;
                found = true;
                break;
            }
        }
        require(found, "Invalid JSON format: missing colon");
        
        // Find the closing quote or comma
        uint endPos = startPos;
        found = false;
        for (uint i = startPos; i < requirements.length; i++) {
            if (requirements[i] == '"' || requirements[i] == ',' || requirements[i] == '}') {
                endPos = i;
                found = true;
                break;
            }
        }
        require(found, "Invalid JSON format: unterminated value");
        
        // Extract the agent type
        bytes memory typeBytes = new bytes(endPos - startPos);
        for (uint i = 0; i < endPos - startPos; i++) {
            typeBytes[i] = requirements[startPos + i];
        }
        
        return string(typeBytes);
    }

    /**
     * @dev Helper function to find index of a substring
     * @param haystack The string to search in
     * @param needle The string to search for
     * @return The index of the substring, or type(uint).max if not found
     */
    function indexOf(bytes memory haystack, bytes memory needle) internal pure returns (uint) {
        if (needle.length == 0) {
            return 0;
        }
        if (haystack.length < needle.length) {
            return type(uint).max;
        }
        
        for (uint i = 0; i <= haystack.length - needle.length; i++) {
            bool found = true;
            for (uint j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return i;
            }
        }
        return type(uint).max;
    }

    /**
     * @dev Helper function to find a numeric value after a key in a JSON string
     */
    function _findValueAfter(bytes memory strBytes, string memory key) internal pure returns (uint256) {
        bytes memory keyBytes = bytes(key);
        uint256 i = 0;
        
        // Find key
        while (i < strBytes.length - keyBytes.length) {
            bool found = true;
            for (uint256 j = 0; j < keyBytes.length; j++) {
                if (strBytes[i + j] != keyBytes[j]) {
                    found = false;
                    break;
                }
            }
            if (found) {
                // Skip to value
                while (i < strBytes.length && strBytes[i] != ':') i++;
                while (i < strBytes.length && (strBytes[i] == ':' || strBytes[i] == ' ')) i++;
                
                // Parse value
                uint256 value = 0;
                while (i < strBytes.length && strBytes[i] >= '0' && strBytes[i] <= '9') {
                    value = value * 10 + uint8(strBytes[i]) - 48;
                    i++;
                }
                return value;
            }
            i++;
        }
        revert("Key not found in JSON");
    }

    /**
     * @dev Helper function to find a string value after a key in a JSON string
     */
    function _findStringValueAfter(bytes memory strBytes, string memory key) internal pure returns (string memory) {
        bytes memory keyBytes = bytes(key);
        uint256 i = 0;
        
        // Find key
        while (i < strBytes.length - keyBytes.length) {
            bool found = true;
            for (uint256 j = 0; j < keyBytes.length; j++) {
                if (strBytes[i + j] != keyBytes[j]) {
                    found = false;
                    break;
                }
            }
            if (found) {
                // Skip to value
                while (i < strBytes.length && strBytes[i] != '"') i++;
                i++; // Skip first quote
                
                // Find end quote
                uint256 start = i;
                while (i < strBytes.length && strBytes[i] != '"') i++;
                
                // Extract string value
                bytes memory valueBytes = new bytes(i - start);
                for (uint256 j = 0; j < i - start; j++) {
                    valueBytes[j] = strBytes[start + j];
                }
                return string(valueBytes);
            }
            i++;
        }
        revert("Key not found in JSON");
    }

    /**
     * @dev Assign an agent to a work unit
     * @param workUnitId The ID of the work unit
     * @param agentId The ID of the agent to assign
     */
    function assignAgent(uint256 workUnitId, bytes32 agentId) 
        public 
        onlyWorkUnitOwner(workUnitId) 
        workUnitExists(workUnitId) 
    {
        WorkUnit storage workUnit = workUnits[workUnitId];
        require(workUnit.isActive, "Work unit is not active");
        require(workUnit.status == WorkUnitStatus.CREATED, "Work unit not in CREATED status");
        require(workUnit.assignedAgentId == bytes32(0), "Work unit already has assigned agent");

        // Parse and verify temporal constraints
        require(verifyTemporalConstraints(workUnit.temporalConstraints), "Temporal constraints not met");

        // Get agent details from registry
        (
            ,  // agentId (unused)
            string memory agentType,
            string[] memory capabilities,
            bool isActive,
            ,  // registeredAt (unused)
            // owner (unused)
        ) = agentRegistry.getAgent(agentId);
        
        require(isActive, "Agent is not active");
        require(capabilities.length > 0, "Agent has no capabilities");

        // Parse and verify agent type compatibility
        string memory requiredType = parseAgentRequirements(workUnit.requirements);
        require(
            keccak256(abi.encodePacked(agentType)) == keccak256(abi.encodePacked(requiredType)),
            "Agent type incompatible"
        );

        workUnit.assignedAgentId = agentId;
        workUnit.status = WorkUnitStatus.ASSIGNED;
        emit WorkUnitAssigned(workUnitId, agentId);
    }

    /**
     * @dev Update work unit status
     * @param workUnitId The ID of the work unit
     * @param newStatus The new status
     * @param statusMetadata Metadata associated with the status change
     */
    function updateStatus(
        uint256 workUnitId,
        WorkUnitStatus newStatus,
        string memory statusMetadata
    ) 
        public 
        workUnitExists(workUnitId)
    {
        WorkUnit storage workUnit = workUnits[workUnitId];
        require(workUnit.isActive, "Work unit is not active");
        
        // Verify status transition is valid
        require(_isValidStatusTransition(workUnit.status, newStatus), "Invalid status transition");
        
        // Store old status for event
        WorkUnitStatus oldStatus = workUnit.status;
        
        // Update status
        workUnit.status = newStatus;
        
        // Emit event with metadata
        emit WorkUnitStatusUpdated(
            workUnitId,
            oldStatus,
            newStatus,
            workUnit.assignedAgentId,
            block.timestamp
        );
    }

    function _isValidStatusTransition(WorkUnitStatus from, WorkUnitStatus to) internal pure returns (bool) {
        if (from == WorkUnitStatus.CREATED) {
            return to == WorkUnitStatus.ASSIGNED || to == WorkUnitStatus.CANCELLED;
        }
        if (from == WorkUnitStatus.ASSIGNED) {
            return to == WorkUnitStatus.IN_PROGRESS || to == WorkUnitStatus.CANCELLED;
        }
        if (from == WorkUnitStatus.IN_PROGRESS) {
            return to == WorkUnitStatus.COMPLETED || to == WorkUnitStatus.FAILED || to == WorkUnitStatus.DISPUTED;
        }
        if (from == WorkUnitStatus.COMPLETED) {
            return to == WorkUnitStatus.VERIFIED || to == WorkUnitStatus.DISPUTED;
        }
        if (from == WorkUnitStatus.DISPUTED) {
            return to == WorkUnitStatus.COMPLETED || to == WorkUnitStatus.FAILED || to == WorkUnitStatus.CANCELLED;
        }
        return false;
    }

    /**
     * @dev Deactivate a work unit
     * @param workUnitId The ID of the work unit
     */
    function deactivateWorkUnit(uint256 workUnitId) 
        public 
        onlyWorkUnitOwner(workUnitId) 
        workUnitExists(workUnitId) 
    {
        require(workUnits[workUnitId].isActive, "Work unit is already inactive");
        workUnits[workUnitId].isActive = false;
        emit WorkUnitDeactivated(workUnitId);
    }

    /**
     * @dev Reactivate a work unit
     * @param workUnitId The ID of the work unit
     */
    function reactivateWorkUnit(uint256 workUnitId) 
        public 
        onlyWorkUnitOwner(workUnitId) 
        workUnitExists(workUnitId) 
    {
        require(!workUnits[workUnitId].isActive, "Work unit is already active");
        workUnits[workUnitId].isActive = true;
        emit WorkUnitReactivated(workUnitId);
    }

    /**
     * @dev Get total number of work units
     * @return Number of work units
     */
    function getWorkUnitCount() public view returns (uint256) {
        return workUnitCounter;
    }

    function verifyTemporalConstraints(string memory temporalConstraintsJson) internal view returns (bool) {
        require(bytes(temporalConstraintsJson).length > 0, "Empty temporal constraints");
        bytes memory constraintsBytes = bytes(temporalConstraintsJson);
        
        // Find start_time field
        uint256 startTimePos = findJsonField(constraintsBytes, "start_time");
        require(startTimePos > 0, "start_time not found");
        
        // Find deadline field
        uint256 deadlinePos = findJsonField(constraintsBytes, "deadline");
        require(deadlinePos > 0, "deadline not found");
        
        // Parse start_time and deadline
        uint256 startTime;
        uint256 endPos;
        (startTime, endPos) = parseUint(constraintsBytes, startTimePos);
        
        uint256 deadline;
        (deadline, endPos) = parseUint(constraintsBytes, deadlinePos);
        
        // Verify temporal constraints
        require(deadline > startTime, "Invalid temporal constraints: deadline must be after start time");
        require(block.timestamp >= startTime, "Work unit has not started yet");
        require(block.timestamp <= deadline, "Work unit deadline has passed");
        
        return true;
    }

    function findJsonField(bytes memory json, string memory fieldName) internal pure returns (uint256) {
        bytes memory searchStr = abi.encodePacked('"', fieldName, '":');
        uint256 searchLen = searchStr.length;
        
        for (uint256 i = 0; i < json.length - searchLen; i++) {
            bool found = true;
            for (uint256 j = 0; j < searchLen; j++) {
                if (json[i + j] != searchStr[j]) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return i + searchLen;
            }
        }
        return 0;
    }

    function parseUint(bytes memory json, uint256 start) internal pure returns (uint256 value, uint256 end) {
        value = 0;
        end = start;
        
        // Skip whitespace
        while (end < json.length && (
            json[end] == 0x20 || // space
            json[end] == 0x09 || // tab
            json[end] == 0x0A || // newline
            json[end] == 0x0D    // carriage return
        )) {
            end++;
        }
        
        // Parse digits
        while (end < json.length && json[end] >= bytes1('0') && json[end] <= bytes1('9')) {
            value = value * 10 + uint8(uint8(json[end]) - uint8(bytes1('0')));
            end++;
        }
        
        require(end > start, "No valid number found");
        return (value, end);
    }
} 