# Agentic Teams

A sophisticated research automation system that leverages specialized AI agents to conduct in-depth research and analysis through Discord. The system uses LangChain, Pydantic, and a state-of-the-art agent architecture to perform complex research tasks.

## Features

- **Specialized Research Agents**: A team of AI agents with different expertise areas working together to conduct research
- **Discord Integration**: Seamless interaction through Discord channels and threads
- **Advanced Research Planning**: Multi-phase research execution with brainstorming, planning, and execution stages
- **Memory Management**: Sophisticated memory system to maintain context and research history
- **Tool Integration**: Various research tools including search, data analysis, graph analysis, and network analysis
- **Blockchain Integration**: Optional blockchain capabilities for task registry and responsibility tracking
- **Benchmarking**: Built-in benchmarking system to track and evaluate research performance
- **Extensive Logging**: Comprehensive logging system with different verbosity levels

## Requirements

```
web3==6.11.3
eth-account==0.8.0
py-solc-x==2.0.2
eth-tester==0.9.1b1
eth-typing==3.5.1
eth-utils==2.3.1
hexbytes==0.3.1
beautifulsoup4==4.12.3
aiohttp==3.9.1
langchain-core==0.1.12
langchain-community==0.0.13
loguru>=0.7.2
langgraph>=0.0.15
rich>=13.7.0
pydantic>=2.5.3
langchain>=0.1.0
```

## Project Structure

- `core/`: Core functionality modules
  - `agents/`: Agent definitions and personas
  - `blockchain/`: Blockchain integration components
  - `compile/`: Task compilation and graph building
  - `engine/`: Main research engine and agent management
  - `memory/`: Memory management system
  - `schemas/`: Pydantic models and JSON schemas
  - `shared/`: Shared utilities and settings
- `tests/`: Test suite
- `diagnostics/`: System diagnostic tools
- `scripts/`: Utility scripts
- `run.py`: Main application entry point
- `run_e2e_test.py`: End-to-end test runner

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file with required configuration
   - Required variables:
     - `DISCORD_BOT_TOKEN`: Your Discord bot token
     - `ETH_PRIVATE_KEY` (optional): For blockchain integration

4. Run the blockchain setup (optional):
   ```bash
   python scripts/setup_blockchain.py
   ```

5. Run the diagnostic tests:
   ```bash
   python diagnostics/phase1_diagnostic.py
   python diagnostics/phase2_diagnostic.py
   python diagnostics/phase3_diagnostic.py
   python diagnostics/phase4_diagnostic.py
   ```

## Usage

1. Start the bot:
   ```bash
   python run.py
   ```

2. In Discord, use the `!research` command followed by your research topic:
   ```
   !research [your research topic]
   ```

3. The system will:
   - Create a new research thread
   - Select appropriate specialized agents
   - Generate a research plan
   - Execute the research steps
   - Provide regular updates and findings

## Architecture

The system uses a sophisticated architecture with several key components:

- **ResearchBot**: Main Discord bot class that handles user interactions
- **ResearchEngine**: Core engine for executing research tasks
- **AgentManager**: Manages specialized research agents
- **MemoryManager**: Handles context and research history
- **GraphBuilder**: Constructs task execution graphs
- **BlockchainManager**: (Optional) Manages blockchain integration

## Development

- Use the provided diagnostic tools to verify system components
- Run end-to-end tests with `python run_e2e_test.py`
- Follow the logging output in `logs/` directory
- Check the benchmark results for performance metrics

## Notes

- All prompt templates use Pydantic output parsers with format='json'
- The system uses LangChain for agent interactions and tool execution
- Rich logging provides detailed execution information
- Blockchain integration is optional but provides additional capabilities

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

