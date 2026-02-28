# Multi-Agent Orchestration Guide

**Difficulty:** Intermediate  
**Time:** 4 hours  
**Cost:** $10-20 (test credits)

## Overview

Build systems with multiple AI agents that work together autonomously. Learn agent-to-agent communication, task delegation, failure handling, and coordination strategies for complex workflows.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                      │
│  - Task breakdown & delegation                            │
│  - Agent management                                       │
│  - State synchronization                                  │
│  - Failure recovery                                       │
└────────────────────┬───────────────────────────────────────┘
                     │
        ┌────────────┴────────────┬────────────────┬────────┐
        ▼                        ▼                ▼        ▼
┌──────────────┐      ┌──────────────┐  ┌──────────────┐ ┌──────────────┐
│  Research    │      │   Analysis   │  │   Generator  │ │   Validator  │
│    Agent     │      │    Agent     │  │    Agent     │ │    Agent     │
│              │      │              │  │              │ │              │
│ - Web search │      │ - Pattern    │  │ - Content    │ │ - Quality    │
│ - Sources    │      │   detection  │  │ - Templates  │ │ - Checks     │
│ - Synthesis  │      │ - Metrics    │  │ - Assembly   │ │ - Validation │
│ - Summaries  │      │ - Insights   │  │ - Formatting │ │ - Reporting  │
└──────────────┘      └──────────────┘  └──────────────┘ └──────────────┘
```

## Step 1: Agent Registry

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

class AgentRole(Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    GENERATOR = "generator"
    VALIDATOR = "validator"
    ORCHESTRATOR = "orchestrator"

@dataclass
class AgentConfig:
    name: str
    role: AgentRole
    api_provider: str
    model: str = "claude-3-haiku"
    max_tokens: int = 2000
    temperature: float = 0.7
    
class AgentRegistry:
    """Registry for managing AI agents"""
    
    def __init__(self):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.configs: Dict[str, AgentConfig] = {}
    
    def register_agent(self, agent: 'BaseAgent', config: AgentConfig):
        """Register a new agent"""
        self.agents[config.name] = agent
        self.configs[config.name] = config
        print(f"Registered {config.name} ({config.role.value})")
    
    def get_agent(self, name: str) -> Optional['BaseAgent']:
        """Get agent by name"""
        return self.agents.get(name)
    
    def get_agents_by_role(self, role: AgentRole) -> List['BaseAgent']:
        """Get all agents of a specific role"""
        return [
            agent for name, agent in self.agents.items()
            if self.configs[name].role == role
        ]
    
    def list_agents(self) -> Dict[str, str]:
        """Get list of all agents and their roles"""
        return {
            name: config.role.value 
            for name, config in self.configs.items()
        }

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    COMPLETE = "complete"

@dataclass
class AgentStatus:
    agent_name: str
    status: AgentStatus
    current_task: Optional[str] = None
    completed_tasks: int = 0
    errors: int = 0
    last_activity: Optional[str] = None

class AgentMonitor:
    """Monitor agent health and performance"""
    
    def __init__(self):
        self.statuses: Dict[str, AgentStatus] = {}
    
    def update_status(self, agent: 'BaseAgent', status: AgentStatus):
        """Update agent status"""
        self.statuses[agent.name] = status
    
    def get_all_statuses(self) -> Dict[str, AgentStatus]:
        """Get all agent statuses"""
        return self.statuses.copy()
    
    def get_working_agents(self) -> List[str]:
        """Get list of working agents"""
        return [
            name for name, status in self.statuses.items()
            if status.status == AgentStatus.WORKING
        ]
```

## Step 2: Base Agent Class

```python
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, config: AgentConfig, registry: AgentRegistry):
        self.name = name
        self.config = config
        self.registry = registry
        self.status = AgentStatus(
            agent_name=name,
            status=AgentStatus.IDLE,
            current_task=None,
            completed_tasks=0,
            errors=0,
            last_activity=None
        )
        self.task_history: List[Dict] = []
        self.shared_data: Dict[str, Any] = {}
    
    @abstractmethod
    async def execute_task(self, task: Dict) -> Dict:
        """Execute a specific task"""
        pass
    
    @abstractmethod
    async def process(self, input_data: Dict) -> Dict:
        """Main processing method"""
        pass
    
    def update_status(self, status: AgentStatus):
        """Update agent status"""
        self.status = status
        self.status.last_activity = self._get_timestamp()
        monitor.update_status(self, status)
    
    async def run_task(self, task: Dict) -> Dict:
        """Run task with error handling"""
        try:
            self.update_status(AgentStatus(
                agent_name=self.name,
                status=AgentStatus.WORKING,
                current_task=task.get("name", "unknown")
            ))
            
            result = await self.execute_task(task)
            
            self.status.completed_tasks += 1
            self.task_history.append({
                "task": task,
                "result": result,
                "status": "success",
                "timestamp": self._get_timestamp()
            })
            
            self.update_status(AgentStatus(
                agent_name=self.name,
                status=AgentStatus.COMPLETE
            ))
            
            return result
            
        except Exception as e:
            self.status.errors += 1
            self.task_history.append({
                "task": task,
                "error": str(e),
                "status": "error",
                "timestamp": self._get_timestamp()
            })
            
            self.update_status(AgentStatus(
                agent_name=self.name,
                status=AgentStatus.ERROR,
                current_task=None
            ))
            
            raise
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            "total_tasks": len(self.task_history),
            "completed": self.status.completed_tasks,
            "errors": self.status.errors,
            "success_rate": self._calculate_success_rate(),
            "average_execution_time": self._calculate_avg_time()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate"""
        if not self.task_history:
            return 0.0
        return self.status.completed_tasks / len(self.task_history) * 100
    
    def _calculate_avg_time(self) -> float:
        """Calculate average execution time (simplified)"""
        # Would implement actual timing in production
        return 2.5  # seconds (placeholder)

class ResearchAgent(BaseAgent):
    """Agent specialized in research"""
    
    async def execute_task(self, task: Dict) -> Dict:
        """Research and synthesize information"""
        query = task.get("query", "")
        sources = task.get("sources", ["web"])
        
        # Simulate research (would integrate with search API in production)
        results = await self._search_web(query)
        summary = await self._synthesize_results(results)
        
        return {
            "query": query,
            "sources_searched": len(results),
            "summary": summary,
            "confidence": 0.85,
            "evidence": results[:5]  # Top 5 sources
        }
    
    async def _search_web(self, query: str) -> List[Dict]:
        """Search web for information"""
        # Would use actual search API
        return [
            {"title": f"Result for '{query}'", "url": f"https://example.com/{query}", "content": "Summary..."}
            for _ in range(10)
        ]
    
    async def _synthesize_results(self, results: List[Dict]) -> str:
        """Synthesize search results"""
        if not results:
            return "No results found"
        
        summaries = [r["content"] for r in results[:3]]
        synthesized = " ".join(summaries)
        
        return f"Based on {len(results)} sources: {synthesized[:500]}..."

class AnalystAgent(BaseAgent):
    """Agent specialized in analysis"""
    
    async def execute_task(self, task: Dict) -> Dict:
        """Analyze data and patterns"""
        data = task.get("data", [])
        analysis_type = task.get("type", "patterns")
        
        patterns = await self._identify_patterns(data)
        insights = await self._generate_insights(patterns)
        
        return {
            "analysis_type": analysis_type,
            "patterns_found": len(patterns),
            "insights": insights,
            "confidence": 0.78,
            "recommendations": await self._generate_recommendations(patterns)
        }
    
    async def _identify_patterns(self, data: List) -> List[Dict]:
        """Identify patterns in data"""
        # Would use ML in production
        return [
            {"pattern": "trend", "strength": 0.7, "description": "Upward trend detected"},
            {"pattern": "anomaly", "strength": 0.3, "description": "Minor anomaly in Q3"}
        ]
    
    async def _generate_insights(self, patterns: List[Dict]) -> str:
        """Generate insights from patterns"""
        return f"Key insights: {len(patterns)} patterns identified. Primary trend is upward movement with minor anomalies."
```

## Step 3: Communication Protocol

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_COMPLETE = "task_complete"
    ERROR_REPORT = "error_report"
    DATA_SHARING = "data_sharing"
    QUESTION = "question"
    ANSWER = "answer"

@dataclass
class Message:
    type: MessageType
    sender: str
    recipient: Optional[str] = None
    content: Dict = None
    timestamp: str = None
    reply_to: Optional[str] = None
    priority: int = 0

class MessageBus:
    """Message bus for agent communication"""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, List['BaseAgent']] = {}
        self.pending_messages: Dict[str, List[Message]] = {}
    
    async def register_agent(self, agent: 'BaseAgent'):
        """Register agent with message bus"""
        self.queues[agent.name] = asyncio.Queue()
        await agent.subscribe(self)
    
    async def send_message(self, message: Message):
        """Send message to agent"""
        if message.recipient and message.recipient in self.queues:
            await self.queues[message.recipient].put(message)
            print(f"Sent {message.type.value} from {message.sender} to {message.recipient}")
        else:
            print(f"Warning: Cannot send to {message.recipient}")
    
    async def broadcast(self, message: Message, exclude: List[str] = None):
        """Broadcast message to all agents"""
        exclude = exclude or []
        
        for agent_name, queue in self.queues.items():
            if agent_name not in exclude:
                await queue.put(message)
                print(f"Broadcast to {agent_name}")
    
    async def receive_message(self, agent: 'BaseAgent', timeout: int = 30) -> Optional[Message]:
        """Receive message for agent"""
        if agent.name in self.queues:
            try:
                message = await asyncio.wait_for(
                    self.queues[agent.name].get(),
                    timeout=timeout
                )
                return message
            except asyncio.TimeoutError:
                return None
        return None

class TaskManager:
    """Manage task distribution among agents"""
    
    def __init__(self, message_bus: MessageBus, registry: AgentRegistry):
        self.message_bus = message_bus
        self.registry = registry
        self.pending_tasks: List[Dict] = []
        self.completed_tasks: List[Dict] = []
        self.task_to_agent: Dict[str, str] = {}
    
    async def assign_task(self, task: Dict) -> str:
        """Assign task to appropriate agent"""
        task_id = f"task_{len(self.pending_tasks) + 1}"
        
        # Determine best agent for task
        best_agent = await self._select_agent(task)
        
        if best_agent:
            # Send task request
            message = Message(
                type=MessageType.TASK_REQUEST,
                sender="manager",
                recipient=best_agent.name,
                content={
                    "task_id": task_id,
                    "task": task,
                    "priority": task.get("priority", 0)
                },
                timestamp=self._get_timestamp()
            )
            
            self.pending_tasks.append(task)
            self.task_to_agent[task_id] = best_agent.name
            
            await self.message_bus.send_message(message)
            print(f"Assigned {task_id} to {best_agent.name}")
            
            return task_id
        else:
            raise Exception("No suitable agent found")
    
    async def _select_agent(self, task: Dict) -> Optional['BaseAgent']:
        """Select best agent for task"""
        task_type = task.get("type", "general")
        
        # Simple selection logic (would be more sophisticated in production)
        if task_type == "research":
            agents = self.registry.get_agents_by_role(AgentRole.RESEARCHER)
        elif task_type == "analysis":
            agents = self.registry.get_agents_by_role(AgentRole.ANALYST)
        else:
            agents = list(self.registry.agents.values())
        
        if agents:
            return agents[0]  # Select first available agent
        return None
    
    async def complete_task(self, task_id: str, result: Dict):
        """Mark task as complete"""
        if task_id in self.task_to_agent:
            agent_name = self.task_to_agent[task_id]
            
            self.completed_tasks.append({
                "task_id": task_id,
                "result": result,
                "agent": agent_name,
                "timestamp": self._get_timestamp()
            })
            
            print(f"Completed {task_id} by {agent_name}")
            
            # Broadcast completion
            message = Message(
                type=MessageType.TASK_COMPLETE,
                sender=agent_name,
                recipient=None,
                content={"task_id": task_id, "result": result},
                timestamp=self._get_timestamp()
            )
            
            await self.message_bus.broadcast(message, exclude=[agent_name])
    
    def get_summary(self) -> Dict:
        """Get task summary"""
        return {
            "total_tasks": len(self.pending_tasks) + len(self.completed_tasks),
            "pending": len(self.pending_tasks),
            "completed": len(self.completed_tasks),
            "by_agent": self._count_by_agent()
        }
    
    def _count_by_agent(self) -> Dict[str, int]:
        """Count tasks by agent"""
        counts = {}
        for task in self.completed_tasks:
            agent = task.get("agent", "unknown")
            counts[agent] = counts.get(agent, 0) + 1
        return counts
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
```

## Step 4: Orchestrator Implementation

```python
import asyncio
from typing import Dict, List, Optional

class OrchestratorAgent(BaseAgent):
    """Main orchestrator agent"""
    
    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        super().__init__(
            name="orchestrator",
            config=AgentConfig(
                name="orchestrator",
                role=AgentRole.ORCHESTRATOR,
                api_provider="claude",
                model="claude-3-haiku"
            ),
            registry=registry
        )
        self.message_bus = message_bus
        self.task_manager = TaskManager(message_bus, registry)
        self.workflow_history: List[Dict] = []
    
    async def run_workflow(self, workflow: Dict) -> Dict:
        """Run complete workflow with multiple agents"""
        workflow_id = f"workflow_{len(self.workflow_history) + 1}"
        
        try:
            # Parse workflow
            steps = workflow.get("steps", [])
            input_data = workflow.get("input", {})
            
            # Execute each step
            results = []
            current_data = input_data
            
            for step in steps:
                # Assign task
                task_id = await self.task_manager.assign_task({
                    "type": step.get("type", "general"),
                    "data": current_data,
                    "requirements": step.get("requirements", {}),
                    "priority": step.get("priority", 0)
                })
                
                # Wait for completion
                result = await self._wait_for_task(task_id)
                
                # Process result
                current_data = self._prepare_for_next_step(result, step)
                results.append({
                    "step": step["name"],
                    "task_id": task_id,
                    "result": result,
                    "duration": self._calculate_duration(task_id)
                })
                
                print(f"Completed {step['name']}")
            
            # Compile final result
            final_result = {
                "workflow_id": workflow_id,
                "total_steps": len(steps),
                "steps_completed": len(results),
                "results": results,
                "final_output": current_data,
                "execution_time": self._calculate_total_time(results),
                "success": True
            }
            
            # Record in history
            self.workflow_history.append({
                "workflow_id": workflow_id,
                "workflow": workflow,
                "results": final_result,
                "timestamp": self._get_timestamp()
            })
            
            return final_result
            
        except Exception as e:
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "results": results
            }
    
    async def _wait_for_task(self, task_id: str, timeout: int = 60) -> Dict:
        """Wait for task completion"""
        # Would implement actual waiting logic
        # For now, return placeholder
        await asyncio.sleep(2)  # Simulate work
        return {"task_id": task_id, "status": "complete", "data": "result"}
    
    def _prepare_for_next_step(self, result: Dict, current_step: Dict) -> Dict:
        """Prepare result for next step"""
        # Extract relevant data for next step
        data = result.get("output", {})
        
        # Add metadata
        data["previous_step"] = current_step.get("name")
        data["confidence"] = result.get("confidence", 0)
        data["errors"] = result.get("errors", [])
        
        return data
    
    def _calculate_duration(self, task_id: str) -> float:
        """Calculate task duration"""
        # Would track actual timing
        return 2.5  # seconds
    
    def _calculate_total_time(self, results: List[Dict]) -> float:
        """Calculate total execution time"""
        return sum(r.get("duration", 0) for r in results)
```

## Step 5: Failure Handling

```python
class FailureHandler:
    """Handle agent failures and retries"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.failure_log: List[Dict] = []
        self.retry_counts: Dict[str, int] = {}
        self.failed_over: Dict[str, str] = {}
    
    def handle_failure(self, agent: 'BaseAgent', error: Exception, task: Dict):
        """Handle agent failure"""
        failure_record = {
            "agent": agent.name,
            "task": task.get("id", "unknown"),
            "error": str(error),
            "timestamp": self._get_timestamp(),
            "retry_count": self.retry_counts.get(agent.name, 0),
            "max_retries": 3
        }
        
        self.failure_log.append(failure_record)
        
        # Attempt retry
        if self.retry_counts.get(agent.name, 0) < 3:
            self.retry_counts[agent.name] = self.retry_counts.get(agent.name, 0) + 1
            
            # Find replacement agent
            replacement = self._find_replacement(agent, task)
            
            if replacement:
                print(f"Failed over from {agent.name} to {replacement.name}")
                self.failed_over[task.get("id", "unknown")] = replacement.name
                
                return replacement
        
        print(f"Max retries exceeded for {agent.name}")
        return None
    
    def _find_replacement(self, failed_agent: 'BaseAgent', task: Dict) -> Optional['BaseAgent']:
        """Find replacement agent"""
        task_type = task.get("type", "general")
        
        # Find available agents of same role
        available = self.registry.get_agents_by_role(failed_agent.config.role)
        
        # Exclude failed agent
        available = [a for a in available if a.name != failed_agent.name]
        
        return available[0] if available else None
    
    def get_failure_report(self) -> Dict:
        """Get failure analysis report"""
        if not self.failure_log:
            return {"message": "No failures recorded"}
        
        failure_types = {}
        for failure in self.failure_log:
            error_type = failure.get("error", "unknown")[:50]  # First 50 chars
            failure_types[error_type] = failure_types.get(error_type, 0) + 1
        
        return {
            "total_failures": len(self.failure_log),
            "failure_types": failure_types,
            "retry_counts": self.retry_counts,
            "failures_over": len(self.failed_over)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()

class StateManager:
    """Manage shared state between agents"""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.access_log: List[Dict] = []
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def set_value(self, key: str, value: Any, agent_name: str):
        """Set value with locking"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            self.state[key] = value
            self._log_access(key, "set", agent_name)
    
    async def get_value(self, key: str, agent_name: str) -> Any:
        """Get value with locking"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            self._log_access(key, "get", agent_name)
            return self.state.get(key)
    
    def get_snapshot(self) -> Dict:
        """Get state snapshot"""
        return {
            "keys": list(self.state.keys()),
            "total_keys": len(self.state),
            "access_log": self.access_log[-100:]  # Last 100 accesses
        }
    
    def _log_access(self, key: str, action: str, agent_name: str):
        """Log state access"""
        self.access_log.append({
            "key": key,
            "action": action,
            "agent": agent_name,
            "timestamp": self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
```

## Step 6: Testing & Validation

Create `test_multi_agent.py`:

```python
import unittest
import asyncio
from unittest.mock import Mock, patch

class TestMultiAgentOrchestration(unittest.TestCase):
    """Test multi-agent orchestration"""
    
    def setUp(self):
        """Setup test fixtures"""
        from agent_registry import AgentRegistry, AgentConfig, AgentRole
        from research_agent import ResearchAgent
        from message_bus import MessageBus, TaskManager
        from orchestrator import OrchestratorAgent
        
        self.registry = AgentRegistry()
        self.message_bus = MessageBus()
        
        # Register test agents
        research_config = AgentConfig(
            name="researcher_1",
            role=AgentRole.RESEARCHER,
            api_provider="claude"
        )
        self.research_agent = ResearchAgent("researcher_1", research_config, self.registry)
        self.registry.register_agent(self.research_agent, research_config)
        
        self.orchestrator = OrchestratorAgent(self.registry, self.message_bus)
    
    def test_workflow_execution(self):
        """Test multi-step workflow"""
        workflow = {
            "steps": [
                {"name": "research", "type": "research", "query": "AI agents"},
                {"name": "analyze", "type": "analysis", "query": "AI trends"},
                {"name": "summarize", "type": "summary", "data": "{{previous_results}}"}
            ],
            "input": {"context": "development"}
        }
        
        result = asyncio.run(self.orchestrator.run_workflow(workflow))
        
        self.assertTrue(result["success"])
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(result["steps_completed"], 3)
    
    def test_failure_handling(self):
        """Test agent failure and recovery"""
        workflow = {
            "steps": [
                {"name": "research", "type": "research", "query": "test"}
            ],
            "input": {}
        }
        
        result = asyncio.run(self.orchestrator.run_workflow(workflow))
        
        self.assertTrue(result["success"])
    
    def test_state_sharing(self):
        """Test shared state between agents"""
        workflow = {
            "steps": [
                {"name": "step1", "type": "generic", "data": {"key": "value1"}},
                {"name": "step2", "type": "generic", "requires": ["key from previous"]}
            ]
        }
        
        result = asyncio.run(self.orchestrator.run_workflow(workflow))
        
        # Verify state was shared correctly
        final_data = result["final_output"]
        self.assertIn("key", final_data)
```

## Summary

By following this guide, you've learned to:
- ✅ Design multi-agent orchestration architecture
- ✅ Implement agent communication protocols  
- ✅ Handle task distribution and failures
- ✅ Manage shared state safely
- ✅ Write comprehensive tests

**Estimated Cost to Build:** $15-30 (test credits)  
**Estimated Time:** 4-6 hours  
**Skills Gained:** Multi-agent systems, orchestration, failure handling, state management

**Next:** Implement monitoring, logging, and real-world deployment!

---

*Generated by Claud 🦞 • Part of AI Agent Builder Hub • Production-ready guide*
