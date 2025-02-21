// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

contract TaskRegistry is Ownable {
    using Strings for uint256;
    
    struct Task {
        bytes32 taskId;
        string description;
        uint256 complexityScore;
        uint256 agentId;
        string ipfsHash;
        TaskStatus status;
        uint256 startTime;
        uint256 endTime;
        TaskMetrics metrics;
        bool exists;
    }
    
    struct TaskMetrics {
        uint256 completionScore;  // 0-100
        uint256 qualityScore;     // 0-100
        uint256 timeEfficiency;   // 0-100
        uint256 toolsUsed;
        uint256 stepsExecuted;
        string metricsIpfsHash;   // Detailed metrics stored in IPFS
    }
    
    enum TaskStatus {
        Pending,
        InProgress,
        Completed,
        Failed
    }
    
    // Events
    event TaskRegistered(bytes32 indexed taskId, uint256 complexityScore, uint256 agentId);
    event TaskStarted(bytes32 indexed taskId, uint256 startTime);
    event TaskCompleted(bytes32 indexed taskId, uint256 endTime, TaskMetrics metrics);
    event TaskFailed(bytes32 indexed taskId, string reason);
    event MetricsUpdated(bytes32 indexed taskId, TaskMetrics metrics);
    
    // State variables
    mapping(bytes32 => Task) public tasks;
    mapping(uint256 => bytes32[]) public agentTasks;  // Agent ID => Task IDs
    uint256 public totalTasks;
    
    // Constants for metric calculations
    uint256 constant COMPLETION_WEIGHT = 40;
    uint256 constant QUALITY_WEIGHT = 40;
    uint256 constant TIME_WEIGHT = 20;
    
    constructor() Ownable(msg.sender) {}
    
    function registerTask(
        string memory description,
        uint256 complexityScore,
        uint256 agentId,
        string memory ipfsHash
    ) public returns (bytes32) {
        require(complexityScore > 0 && complexityScore <= 10, "Invalid complexity score");
        
        bytes32 taskId = keccak256(abi.encodePacked(
            block.timestamp,
            description,
            complexityScore,
            agentId
        ));
        
        require(!tasks[taskId].exists, "Task already exists");
        
        tasks[taskId] = Task({
            taskId: taskId,
            description: description,
            complexityScore: complexityScore,
            agentId: agentId,
            ipfsHash: ipfsHash,
            status: TaskStatus.Pending,
            startTime: 0,
            endTime: 0,
            metrics: TaskMetrics({
                completionScore: 0,
                qualityScore: 0,
                timeEfficiency: 0,
                toolsUsed: 0,
                stepsExecuted: 0,
                metricsIpfsHash: ""
            }),
            exists: true
        });
        
        agentTasks[agentId].push(taskId);
        totalTasks++;
        
        emit TaskRegistered(taskId, complexityScore, agentId);
        return taskId;
    }
    
    function startTask(bytes32 taskId) public {
        require(tasks[taskId].exists, "Task does not exist");
        require(tasks[taskId].status == TaskStatus.Pending, "Task not in pending state");
        
        tasks[taskId].status = TaskStatus.InProgress;
        tasks[taskId].startTime = block.timestamp;
        
        emit TaskStarted(taskId, block.timestamp);
    }
    
    function completeTask(
        bytes32 taskId,
        uint256 completionScore,
        uint256 qualityScore,
        uint256 timeEfficiency,
        uint256 toolsUsed,
        uint256 stepsExecuted,
        string memory metricsIpfsHash
    ) public {
        require(tasks[taskId].exists, "Task does not exist");
        require(tasks[taskId].status == TaskStatus.InProgress, "Task not in progress");
        require(completionScore <= 100 && qualityScore <= 100 && timeEfficiency <= 100,
                "Invalid metric scores");
        
        Task storage task = tasks[taskId];
        task.status = TaskStatus.Completed;
        task.endTime = block.timestamp;
        
        TaskMetrics memory metrics = TaskMetrics({
            completionScore: completionScore,
            qualityScore: qualityScore,
            timeEfficiency: timeEfficiency,
            toolsUsed: toolsUsed,
            stepsExecuted: stepsExecuted,
            metricsIpfsHash: metricsIpfsHash
        });
        
        task.metrics = metrics;
        
        emit TaskCompleted(taskId, block.timestamp, metrics);
        emit MetricsUpdated(taskId, metrics);
    }
    
    function failTask(bytes32 taskId, string memory reason) public {
        require(tasks[taskId].exists, "Task does not exist");
        require(tasks[taskId].status == TaskStatus.InProgress, "Task not in progress");
        
        tasks[taskId].status = TaskStatus.Failed;
        tasks[taskId].endTime = block.timestamp;
        
        emit TaskFailed(taskId, reason);
    }
    
    function calculateTaskScore(bytes32 taskId) public view returns (uint256) {
        require(tasks[taskId].exists, "Task does not exist");
        require(tasks[taskId].status == TaskStatus.Completed, "Task not completed");
        
        TaskMetrics memory metrics = tasks[taskId].metrics;
        
        // Calculate weighted score
        uint256 weightedScore = (
            metrics.completionScore * COMPLETION_WEIGHT +
            metrics.qualityScore * QUALITY_WEIGHT +
            metrics.timeEfficiency * TIME_WEIGHT
        ) / 100;
        
        // Scale by complexity
        return (weightedScore * tasks[taskId].complexityScore) / 10;
    }
    
    function getTask(bytes32 taskId) public view returns (
        string memory description,
        uint256 complexityScore,
        uint256 agentId,
        string memory ipfsHash,
        TaskStatus status,
        uint256 startTime,
        uint256 endTime,
        TaskMetrics memory metrics
    ) {
        require(tasks[taskId].exists, "Task does not exist");
        Task memory task = tasks[taskId];
        
        return (
            task.description,
            task.complexityScore,
            task.agentId,
            task.ipfsHash,
            task.status,
            task.startTime,
            task.endTime,
            task.metrics
        );
    }
    
    function getAgentTasks(uint256 agentId) public view returns (bytes32[] memory) {
        return agentTasks[agentId];
    }
    
    function getAgentTaskCount(uint256 agentId) public view returns (uint256) {
        return agentTasks[agentId].length;
    }
    
    function getAgentSuccessRate(uint256 agentId) public view returns (uint256) {
        bytes32[] memory taskIds = agentTasks[agentId];
        if (taskIds.length == 0) return 0;
        
        uint256 completedTasks = 0;
        uint256 totalScore = 0;
        
        for (uint256 i = 0; i < taskIds.length; i++) {
            if (tasks[taskIds[i]].status == TaskStatus.Completed) {
                completedTasks++;
                totalScore += calculateTaskScore(taskIds[i]);
            }
        }
        
        if (completedTasks == 0) return 0;
        return (totalScore * 100) / (completedTasks * 100);  // Normalize to 0-100
    }
} 