// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ResponsibilityNFT
 * @dev Simplified NFT contract for tracking responsibilities in the ATR Framework
 */
contract ResponsibilityNFT {
    // Token name and symbol
    string public name = "ATR Responsibility";
    string public symbol = "ATRR";
    
    // Token ID counter
    uint256 private _nextTokenId = 1;
    
    // Ownership mapping
    mapping(uint256 => address) private _owners;
    mapping(address => uint256) private _balances;
    
    // Responsibility metadata
    struct ResponsibilityMetadata {
        bytes32 agentId;
        uint256 workUnitId;
        string actionType;  // e.g., "assignment", "completion", "verification"
        uint256 timestamp;
        string metadataURI;  // Points to off-chain metadata
    }
    
    // Token metadata mapping
    mapping(uint256 => ResponsibilityMetadata) public responsibilities;
    
    // Events
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event ResponsibilityMinted(
        uint256 indexed tokenId,
        bytes32 indexed agentId,
        uint256 indexed workUnitId,
        string actionType
    );
    
    /**
     * @dev Returns the owner of the token
     */
    function ownerOf(uint256 tokenId) public view returns (address) {
        address owner = _owners[tokenId];
        require(owner != address(0), "Token does not exist");
        return owner;
    }
    
    /**
     * @dev Returns the number of tokens owned by an address
     */
    function balanceOf(address owner) public view returns (uint256) {
        require(owner != address(0), "Invalid address");
        return _balances[owner];
    }
    
    /**
     * @dev Mints a new responsibility NFT
     */
    function mintResponsibility(
        bytes32 agentId,
        uint256 workUnitId,
        string memory actionType,
        string memory metadataURI
    ) public returns (uint256) {
        require(agentId != bytes32(0), "Invalid agent ID");
        require(bytes(actionType).length > 0, "Invalid action type");
        require(bytes(metadataURI).length > 0, "Invalid metadata URI");
        
        uint256 tokenId = _nextTokenId++;
        
        _owners[tokenId] = msg.sender;
        _balances[msg.sender] += 1;
        
        responsibilities[tokenId] = ResponsibilityMetadata({
            agentId: agentId,
            workUnitId: workUnitId,
            actionType: actionType,
            timestamp: block.timestamp,
            metadataURI: metadataURI
        });
        
        emit Transfer(address(0), msg.sender, tokenId);
        emit ResponsibilityMinted(tokenId, agentId, workUnitId, actionType);
        
        return tokenId;
    }
    
    /**
     * @dev Transfers a token
     */
    function transfer(address to, uint256 tokenId) public {
        require(to != address(0), "Invalid recipient");
        require(_owners[tokenId] == msg.sender, "Not the token owner");
        
        _balances[msg.sender] -= 1;
        _balances[to] += 1;
        _owners[tokenId] = to;
        
        emit Transfer(msg.sender, to, tokenId);
    }
    
    /**
     * @dev Gets responsibility metadata
     */
    function getResponsibility(uint256 tokenId) public view returns (
        bytes32 agentId,
        uint256 workUnitId,
        string memory actionType,
        uint256 timestamp,
        string memory metadataURI
    ) {
        require(_owners[tokenId] != address(0), "Token does not exist");
        ResponsibilityMetadata memory metadata = responsibilities[tokenId];
        return (
            metadata.agentId,
            metadata.workUnitId,
            metadata.actionType,
            metadata.timestamp,
            metadata.metadataURI
        );
    }
} 