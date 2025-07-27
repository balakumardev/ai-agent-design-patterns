#!/usr/bin/env python3
"""
Example usage of the RAG-Enhanced Agent Pattern.

This script demonstrates how to use the RAGEnhancedAgent to create
a knowledge-based AI assistant that can answer questions using
retrieved document context.
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from rag_agent import RAGEnhancedAgent
from document_processor import create_sample_documents
from vector_store import VectorStoreConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

# Load environment variables
load_dotenv()

console = Console()


def create_knowledge_base_demo():
    """Create a comprehensive knowledge base for demonstration."""
    
    # Create temporary directory for demo documents
    demo_docs_dir = "demo_knowledge_base"
    Path(demo_docs_dir).mkdir(exist_ok=True)
    
    # Create comprehensive demo documents
    documents = {
        "python_programming.txt": """
Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support
- Versatile applications (web, data science, AI, automation)

Data Types:
1. Numbers (int, float, complex)
2. Strings (text data)
3. Lists (ordered, mutable collections)
4. Tuples (ordered, immutable collections)
5. Dictionaries (key-value pairs)
6. Sets (unordered, unique elements)

Control Structures:
- if/elif/else statements
- for and while loops
- try/except error handling
- with statements for context management

Popular Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Requests: HTTP library
- Django/Flask: Web frameworks
- TensorFlow/PyTorch: Machine learning
        """,
        
        "web_development.txt": """
Modern Web Development

Web development involves creating websites and web applications using various technologies.

Frontend Technologies:
1. HTML (HyperText Markup Language)
   - Structure and content of web pages
   - Semantic elements for accessibility
   - Forms and multimedia integration

2. CSS (Cascading Style Sheets)
   - Styling and layout
   - Responsive design with media queries
   - Flexbox and Grid layouts
   - Animations and transitions

3. JavaScript
   - Interactive functionality
   - DOM manipulation
   - Event handling
   - Asynchronous programming (Promises, async/await)

Frontend Frameworks:
- React: Component-based library by Facebook
- Vue.js: Progressive framework
- Angular: Full-featured framework by Google
- Svelte: Compile-time optimized framework

Backend Technologies:
- Node.js: JavaScript runtime for server-side
- Python: Django, Flask, FastAPI
- Java: Spring Boot
- C#: ASP.NET Core
- PHP: Laravel, Symfony
- Ruby: Ruby on Rails

Databases:
- SQL: PostgreSQL, MySQL, SQLite
- NoSQL: MongoDB, Redis, Cassandra

Development Tools:
- Version Control: Git, GitHub, GitLab
- Package Managers: npm, yarn, pip
- Build Tools: Webpack, Vite, Parcel
- Testing: Jest, Cypress, Selenium
        """,
        
        "data_science.txt": """
Data Science Fundamentals

Data science combines statistics, programming, and domain expertise to extract insights from data.

Data Science Process:
1. Problem Definition
   - Understand business objectives
   - Define success metrics
   - Identify data requirements

2. Data Collection
   - Gather relevant datasets
   - APIs, databases, web scraping
   - Ensure data quality and completeness

3. Data Cleaning and Preprocessing
   - Handle missing values
   - Remove duplicates and outliers
   - Data type conversions
   - Feature engineering

4. Exploratory Data Analysis (EDA)
   - Statistical summaries
   - Data visualization
   - Pattern identification
   - Hypothesis generation

5. Modeling
   - Algorithm selection
   - Training and validation
   - Hyperparameter tuning
   - Model evaluation

6. Deployment and Monitoring
   - Production deployment
   - Performance monitoring
   - Model maintenance and updates

Key Tools and Libraries:
- Python: pandas, NumPy, scikit-learn, matplotlib, seaborn
- R: dplyr, ggplot2, caret, shiny
- SQL: Data querying and manipulation
- Jupyter Notebooks: Interactive development
- Git: Version control for code and data

Machine Learning Types:
- Supervised Learning: Classification, Regression
- Unsupervised Learning: Clustering, Dimensionality Reduction
- Reinforcement Learning: Decision making through rewards

Visualization Tools:
- Matplotlib and Seaborn (Python)
- ggplot2 (R)
- Tableau and Power BI (Business Intelligence)
- D3.js (Web-based visualizations)
        """,
        
        "cloud_computing.txt": """
Cloud Computing Overview

Cloud computing delivers computing services over the internet, providing scalable and flexible IT resources.

Service Models:
1. Infrastructure as a Service (IaaS)
   - Virtual machines, storage, networks
   - Examples: AWS EC2, Google Compute Engine, Azure VMs
   - Full control over operating systems and applications

2. Platform as a Service (PaaS)
   - Development platforms and tools
   - Examples: Heroku, Google App Engine, Azure App Service
   - Focus on application development without infrastructure management

3. Software as a Service (SaaS)
   - Ready-to-use applications
   - Examples: Gmail, Salesforce, Microsoft 365
   - Access via web browsers or APIs

Deployment Models:
- Public Cloud: Services available to general public
- Private Cloud: Dedicated to single organization
- Hybrid Cloud: Combination of public and private
- Multi-Cloud: Using multiple cloud providers

Major Cloud Providers:
1. Amazon Web Services (AWS)
   - Market leader with comprehensive services
   - EC2, S3, Lambda, RDS, DynamoDB

2. Microsoft Azure
   - Strong integration with Microsoft ecosystem
   - Virtual Machines, Blob Storage, Functions, SQL Database

3. Google Cloud Platform (GCP)
   - Strengths in data analytics and machine learning
   - Compute Engine, Cloud Storage, BigQuery, AI Platform

Benefits:
- Cost efficiency and scalability
- Global accessibility and availability
- Automatic updates and maintenance
- Disaster recovery and backup
- Rapid deployment and innovation

Security Considerations:
- Shared responsibility model
- Data encryption in transit and at rest
- Identity and access management
- Compliance with regulations (GDPR, HIPAA)
- Regular security audits and monitoring
        """
    }
    
    # Write documents to files
    for filename, content in documents.items():
        file_path = Path(demo_docs_dir) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    console.print(f"✅ Created {len(documents)} demo documents in {demo_docs_dir}/", style="green")
    return demo_docs_dir


def main():
    """Run example demonstrations of the RAG-enhanced agent."""
    
    console.print(Panel.fit("RAG-Enhanced Agent Pattern Example", style="bold blue"))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ Please set OPENAI_API_KEY in your .env file", style="bold red")
        console.print("Copy .env.example to .env and add your OpenAI API key")
        return
    
    # Initialize the agent
    console.print("Initializing RAG-enhanced agent...", style="yellow")
    
    # Use a temporary directory for this demo
    with tempfile.TemporaryDirectory() as temp_dir:
        config = VectorStoreConfig(
            persist_directory=temp_dir,
            collection_name="demo_collection",
            similarity_threshold=0.6
        )
        
        agent = RAGEnhancedAgent(vector_store_config=config)
        console.print("✅ Agent ready!", style="green")
        
        # Create and add knowledge base
        console.print("\n📚 Building Knowledge Base", style="bold cyan")
        demo_docs_dir = create_knowledge_base_demo()
        
        try:
            # Add documents to knowledge base
            agent.add_documents_from_directory(demo_docs_dir)
            
            # Display knowledge base statistics
            console.print("\n📊 Knowledge Base Statistics:", style="bold blue")
            agent.get_knowledge_base_stats()
            
            # Example queries that demonstrate RAG capabilities
            examples = [
                {
                    "category": "Programming",
                    "query": "What are the key features of Python programming language?",
                    "expected_context": "Should retrieve information from python_programming.txt"
                },
                {
                    "category": "Web Development",
                    "query": "What are the main frontend frameworks and their characteristics?",
                    "expected_context": "Should retrieve information from web_development.txt"
                },
                {
                    "category": "Data Science",
                    "query": "Explain the data science process and key steps involved",
                    "expected_context": "Should retrieve information from data_science.txt"
                },
                {
                    "category": "Cloud Computing",
                    "query": "What are the different cloud service models and their examples?",
                    "expected_context": "Should retrieve information from cloud_computing.txt"
                },
                {
                    "category": "Cross-Domain",
                    "query": "How can Python be used in data science and what tools are available?",
                    "expected_context": "Should retrieve information from both python_programming.txt and data_science.txt"
                }
            ]
            
            console.print("\n🚀 Running RAG-Enhanced Query Examples", style="bold cyan")
            console.print("Each example shows how the agent retrieves relevant context and generates informed responses.\n")
            
            for i, example in enumerate(examples, 1):
                console.print(f"[bold]Example {i}: {example['category']}[/bold]")
                console.print(f"[dim]Expected behavior: {example['expected_context']}[/dim]")
                console.print(f"[cyan]Query:[/cyan] {example['query']}")
                console.print("-" * 80, style="dim")
                
                # Process the query
                response = agent.query(example['query'])
                
                # Display the response
                console.print(f"\n[green]🤖 Agent Response:[/green]")
                console.print(Panel(response.answer, border_style="green", title="Generated Answer"))
                
                # Display confidence score
                confidence_percent = response.confidence_score * 100
                confidence_color = "green" if confidence_percent >= 70 else "yellow" if confidence_percent >= 50 else "red"
                console.print(f"\n[bold blue]📊 Response Quality:[/bold blue]")
                console.print(f"  Confidence Score: [{confidence_color}]{confidence_percent:.1f}%[/{confidence_color}]")
                
                # Display retrieved sources
                if response.sources:
                    console.print(f"\n[bold blue]📚 Retrieved Sources ({len(response.sources)}):[/bold blue]")
                    
                    sources_table = Table(show_header=True, header_style="bold magenta")
                    sources_table.add_column("Source", style="cyan")
                    sources_table.add_column("Similarity", style="green")
                    sources_table.add_column("Content Preview", style="yellow")
                    
                    for source in response.sources[:3]:  # Show top 3 sources
                        source_name = Path(source.source).name
                        similarity_percent = source.similarity_score * 100
                        content_preview = source.content[:100] + "..." if len(source.content) > 100 else source.content
                        
                        sources_table.add_row(
                            source_name,
                            f"{similarity_percent:.1f}%",
                            content_preview
                        )
                    
                    console.print(sources_table)
                else:
                    console.print("[yellow]No relevant sources found[/yellow]")
                
                # Show RAG benefits demonstrated
                console.print(f"\n[bold green]✨ RAG Benefits Demonstrated:[/bold green]")
                benefits = [
                    f"📖 Knowledge Retrieval: Found {len(response.sources)} relevant document chunks",
                    f"🎯 Context-Aware: Response based on specific domain knowledge",
                    f"📊 Confidence Scoring: {confidence_percent:.1f}% confidence in response quality",
                    f"🔍 Source Attribution: Clear traceability to original documents"
                ]
                
                for benefit in benefits:
                    console.print(f"  {benefit}")
                
                console.print("\n" + "="*80 + "\n")
            
            # Demonstrate knowledge base search
            console.print("[bold magenta]🔍 Knowledge Base Search Demonstration[/bold magenta]")
            console.print("Direct search of the knowledge base without question answering:")
            console.print("-" * 80, style="dim")
            
            search_queries = [
                "machine learning algorithms",
                "JavaScript frameworks",
                "cloud security"
            ]
            
            for query in search_queries:
                console.print(f"\n[cyan]Search Query:[/cyan] {query}")
                agent.search_knowledge_base(query, k=3)
            
            # Interactive demonstration
            console.print("\n[bold magenta]💬 Interactive RAG Assistant[/bold magenta]")
            console.print("Ask questions about programming, web development, data science, or cloud computing!")
            console.print("Type 'search: <query>' to search the knowledge base directly")
            console.print("Type 'stats' to see knowledge base statistics")
            console.print("Type 'quit' to exit")
            console.print("-" * 80, style="dim")
            
            while True:
                try:
                    user_input = console.input("\n[bold cyan]Your question:[/bold cyan] ")
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif user_input.lower() == 'stats':
                        agent.get_knowledge_base_stats()
                        continue
                    elif user_input.lower().startswith('search:'):
                        search_query = user_input[7:].strip()
                        agent.search_knowledge_base(search_query, k=5)
                        continue
                    
                    # Process as RAG query
                    response = agent.query(user_input)
                    
                    console.print(f"\n[green]🤖 Answer:[/green]")
                    console.print(Panel(response.answer, border_style="green"))
                    
                    # Show confidence and source count
                    confidence_percent = response.confidence_score * 100
                    confidence_color = "green" if confidence_percent >= 70 else "yellow" if confidence_percent >= 50 else "red"
                    
                    console.print(f"\n[{confidence_color}]Confidence: {confidence_percent:.1f}%[/{confidence_color}] | "
                                f"[blue]Sources: {len(response.sources)}[/blue]")
                    
                    if response.sources:
                        top_source = response.sources[0]
                        source_name = Path(top_source.source).name
                        console.print(f"[dim]Primary source: {source_name}[/dim]")
                    
                except KeyboardInterrupt:
                    console.print("\n👋 Goodbye!", style="bold yellow")
                    break
                except Exception as e:
                    console.print(f"[red]Error:[/red] {str(e)}")
        
        finally:
            # Clean up demo documents
            import shutil
            if os.path.exists(demo_docs_dir):
                shutil.rmtree(demo_docs_dir)
    
    console.print("\n✨ RAG-Enhanced Agent demonstration completed!", style="bold green")
    console.print("\nKey benefits of RAG-Enhanced Agents:")
    
    benefits = [
        "📚 Access to external knowledge beyond training data",
        "🎯 Context-aware responses based on relevant documents",
        "🔍 Transparent source attribution and traceability",
        "📊 Confidence scoring for response quality assessment",
        "🔄 Dynamic knowledge base updates without retraining",
        "💡 Reduced hallucinations through grounded responses",
        "🎨 Flexible document types and formats support"
    ]
    
    for benefit in benefits:
        console.print(f"  {benefit}")
    
    console.print("\nTo run the full interactive demo, use: python rag_agent.py")


if __name__ == "__main__":
    main()
