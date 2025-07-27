# 🤖 Agent Design Patterns

> **📖 Read the Full Blog Post**: [8 Essential AI Agent Design Patterns Every Developer Should Know](https://your-blog-url.com/agent-design-patterns)

A comprehensive collection of **8 essential AI agent design patterns** implemented with **LangGraph** and modern Python frameworks. This repository provides **practical, runnable examples** of each pattern with clear documentation, working code, and comprehensive tests.

**Perfect for**: AI developers, researchers, and anyone building intelligent agent systems who wants to understand and implement proven design patterns.

## ⚡ Quick Start (30 seconds)

```bash
git clone https://github.com/your-username/agent-design-patterns.git
cd agent-design-patterns
python setup.py  # Automated setup
# Edit .env with your API key, then:
cd 01-tool-using-agent && python example.py
```

## 🎯 What You'll Learn

- **8 Production-Ready Patterns**: From basic tool usage to advanced multi-agent coordination
- **Real Working Code**: Every pattern includes complete, runnable implementations
- **Best Practices**: Industry-standard approaches to agent architecture
- **Testing & Validation**: Comprehensive test suites for each pattern
- **Flexible Integration**: Works with OpenAI, Anthropic, local LLMs, and LiteLLM

## 📋 The 8 Essential Patterns

| Pattern | Description | Use Cases | Complexity |
|---------|-------------|-----------|------------|
| **[🔧 Tool-Using Agent](./01-tool-using-agent/)** | Agents that dynamically select and use external tools | API integration, calculations, file operations | ⭐⭐ |
| **[🔄 Reflection Pattern](./02-reflection-pattern/)** | Self-correcting agents that evaluate and improve outputs | Quality assurance, iterative refinement | ⭐⭐⭐ |
| **[🏗️ Hierarchical Planning](./03-hierarchical-planning/)** | Multi-level task decomposition and execution | Project management, complex workflows | ⭐⭐⭐⭐ |
| **[🤝 Multi-Agent Coordination](./04-multi-agent-coordination/)** | Collaborative specialized agent systems | Team automation, distributed problem solving | ⭐⭐⭐⭐⭐ |
| **[📚 RAG-Enhanced Agent](./05-rag-enhanced-agent/)** | Retrieval-augmented generation with vector search | Knowledge bases, document Q&A | ⭐⭐⭐ |
| **[🔄 State Machine Pattern](./06-state-machine-pattern/)** | Finite state machines for controlled behavior | Workflows, conversation management | ⭐⭐⭐ |
| **[🛡️ Circuit Breaker Pattern](./07-circuit-breaker-pattern/)** | Fault tolerance and resilience patterns | Production systems, error handling | ⭐⭐⭐⭐ |
| **[⚖️ Constitutional AI](./08-constitutional-ai-pattern/)** | Rule-based governance and safety measures | Content moderation, ethical AI | ⭐⭐⭐⭐ |

## 🚀 Quick Start (2 Minutes)

### Option 1: Automated Setup (Recommended)
```bash
# Clone and setup everything automatically
git clone https://github.com/your-username/agent-design-patterns.git
cd agent-design-patterns
python setup.py
```

### Option 2: Manual Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-username/agent-design-patterns.git
cd agent-design-patterns

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API access
cp .env.example .env
# Edit .env with your API keys (see configuration section below)
```

### 🔑 API Configuration

Choose one of these options in your `.env` file:

**Option A: OpenAI (Easiest)**
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

**Option B: Local LLM via LiteLLM**
```bash
OPENAI_API_KEY=your_litellm_key
OPENAI_BASE_URL=http://localhost:4000
```

**Option C: Anthropic Claude**
```bash
ANTHROPIC_API_KEY=your_anthropic_key
```

## ▶️ Running the Patterns

### 🎮 Interactive Demos
Each pattern includes an interactive demo:

```bash
# Start with the simplest pattern
cd 01-tool-using-agent
python example.py

# Try the reflection pattern
cd ../02-reflection-pattern
python example.py

# Explore RAG capabilities
cd ../05-rag-enhanced-agent
python example.py
```

### 🧪 Run Tests
Validate everything works:

```bash
# Test a specific pattern
cd 01-tool-using-agent
python -m pytest test_tool_agent.py -v

# Test all patterns
python -m pytest -v
```

### 🔍 What Each Demo Shows

- **Tool-Using Agent**: Calculator, file operations, web search
- **Reflection Pattern**: Self-correction and iterative improvement
- **Hierarchical Planning**: Complex task breakdown and execution
- **Multi-Agent Coordination**: Specialized agents working together
- **RAG-Enhanced Agent**: Document search and knowledge retrieval
- **State Machine Pattern**: Conversation flow management
- **Circuit Breaker Pattern**: Fault tolerance and recovery
- **Constitutional AI**: Safety and governance controls

## 🛠️ Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **LangGraph** | 0.5.4+ | State machine framework for AI agents |
| **LangChain** | 0.3.27+ | LLM application framework |
| **Python** | 3.9+ | Programming language |
| **ChromaDB** | 0.5.5+ | Vector database for RAG pattern |
| **HuggingFace** | Latest | Local embeddings (no API required) |
| **Tenacity** | 8.5.0+ | Retry logic and resilience |
| **Rich** | 13.7.1+ | Beautiful console output |
| **Pydantic** | 2.0+ | Data validation and settings |

### 🔌 Supported LLM Providers

- ✅ **OpenAI** (GPT-4, GPT-4o, GPT-3.5)
- ✅ **Anthropic** (Claude 3.5 Sonnet, Claude 3 Haiku)
- ✅ **Local Models** (via LiteLLM, Ollama)
- ✅ **Azure OpenAI**
- ✅ **Google Gemini**
- ✅ **Any OpenAI-compatible API**

## 📚 Detailed Pattern Guide

### 🔧 1. Tool-Using Agent
**What it does**: Dynamically selects and uses external tools (calculator, file system, web search)
**Key Features**:
- Tool discovery and selection
- Dynamic function calling
- Error handling and retries
- 13 comprehensive tests included

**Example Use Cases**: API integrations, data processing, web scraping

### 🔄 2. Reflection Pattern
**What it does**: Self-evaluates outputs and iteratively improves responses
**Key Features**:
- Self-critique mechanisms
- Iterative refinement loops
- Quality scoring
- Improvement tracking

**Example Use Cases**: Content generation, code review, quality assurance

### 🏗️ 3. Hierarchical Planning
**What it does**: Breaks complex tasks into manageable subtasks with coordination
**Key Features**:
- Multi-level task decomposition
- Dependency management
- Progress tracking
- Dynamic replanning

**Example Use Cases**: Project management, workflow automation, complex problem solving

### 🤝 4. Multi-Agent Coordination
**What it does**: Coordinates multiple specialized agents to solve complex problems
**Key Features**:
- Agent specialization
- Task distribution
- Communication protocols
- Result aggregation

**Example Use Cases**: Team automation, distributed systems, collaborative AI

### 📚 5. RAG-Enhanced Agent
**What it does**: Combines retrieval with generation for knowledge-based responses
**Key Features**:
- Document ingestion and chunking
- Vector similarity search
- Local embeddings (no API required)
- Source attribution

**Example Use Cases**: Knowledge bases, document Q&A, research assistance

### 🔄 6. State Machine Pattern
**What it does**: Controls agent behavior through defined states and transitions
**Key Features**:
- Finite state machines
- Transition logic
- State persistence
- Event handling

**Example Use Cases**: Conversation flows, workflow management, game AI

### 🛡️ 7. Circuit Breaker Pattern
**What it does**: Provides fault tolerance and graceful degradation
**Key Features**:
- Failure detection
- Automatic recovery
- Fallback mechanisms
- Performance monitoring

**Example Use Cases**: Production systems, API reliability, error handling

### ⚖️ 8. Constitutional AI Pattern
**What it does**: Ensures AI behavior aligns with predefined ethical principles
**Key Features**:
- Rule-based governance
- Safety evaluations
- Violation detection
- Transparent assessments

**Example Use Cases**: Content moderation, ethical AI, compliance systems

## 🔧 Troubleshooting

### Common Issues

**❌ "Module not found" errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**❌ "API key not found" errors**
```bash
# Check your .env file exists and has the right format
cp .env.example .env
# Edit .env with your actual API keys
```

**❌ "Model not found" errors**
```bash
# For LiteLLM users, check your model names match your setup
# For OpenAI users, verify your API key has access to the models
```

**❌ RAG pattern fails to load embeddings**
```bash
# The pattern uses local HuggingFace models - first run downloads them
# Ensure you have internet connection for the initial download
```

### Getting Help

1. **Check the Logs**: Each pattern provides detailed error messages
2. **Review Examples**: Look at `example.py` in each directory
3. **Run Tests**: Use `pytest` to validate your setup
4. **Check Documentation**: Each pattern has detailed README.md
5. **Open an Issue**: For bugs or questions not covered here

## 📁 Repository Structure

```
agent-design-patterns/
├── 01-tool-using-agent/          # Basic tool usage
├── 02-reflection-pattern/         # Self-improvement
├── 03-hierarchical-planning/      # Task decomposition
├── 04-multi-agent-coordination/   # Agent collaboration
├── 05-rag-enhanced-agent/         # Knowledge retrieval
├── 06-state-machine-pattern/      # State management
├── 07-circuit-breaker-pattern/    # Fault tolerance
├── 08-constitutional-ai-pattern/  # AI governance
├── .env.example                   # Configuration template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Global dependencies
├── setup.py                       # Automated setup script
└── README.md                      # This file
```

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-pattern`)
3. **Add** your improvements with tests
4. **Commit** your changes (`git commit -m 'Add amazing pattern'`)
5. **Push** to the branch (`git push origin feature/amazing-pattern`)
6. **Open** a Pull Request

### Contribution Guidelines
- Include comprehensive tests for new patterns
- Follow the existing code style and structure
- Update documentation for any changes
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

If you find this repository helpful, please consider giving it a star! ⭐

## 📞 Support & Community

- **📖 Blog Post**: [8 Essential AI Agent Design Patterns](https://your-blog-url.com/agent-design-patterns)
- **🐛 Issues**: [GitHub Issues](https://github.com/your-username/agent-design-patterns/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-username/agent-design-patterns/discussions)
- **📧 Contact**: [your-email@example.com](mailto:your-email@example.com)

---

**Made with ❤️ for the AI developer community**