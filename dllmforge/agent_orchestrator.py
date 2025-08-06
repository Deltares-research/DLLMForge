"""
Data Analysis Agent System using Google's Agent Development Kit (ADK)
Phase 2-3 Implementation: Core Agent Structure and Specialized Tools

This module implements a multi-agent system for data analysis that helps non-technical
users query databases and perform sophisticated data analysis through natural language.
"""
#TODO: legacy code, change from pandasai import SmartDataframe to pai

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
import warnings

# Google ADK imports
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Data Analysis Agent System - Libraries imported successfully.")

# Environment configuration
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')
LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')

# Database connection configurations
DATABASE_CONFIGS = {
    'sales_db': os.environ.get('SALES_DB_URL'),
    'customer_db': os.environ.get('CUSTOMER_DB_URL'),
    'product_db': os.environ.get('PRODUCT_DB_URL')
}

# =============================================================================
# PHASE 2: SPECIALIZED TOOLS IMPLEMENTATION WITH PANDASAI
# =============================================================================

# PandasAI imports
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Initialize PandasAI LLM
def get_pandasai_llm():
    """Initialize PandasAI LLM with OpenAI"""
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY or OPENAI_API_KEY environment variable")
    return OpenAI(api_token=api_key)

def load_csv_data_tool(file_path: str) -> Dict[str, Any]:
    """
    Loads CSV data and creates a SmartDataframe with PandasAI.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing loaded data info and SmartDataframe
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "available_files": [f for f in os.listdir('.') if f.endswith('.csv')]
            }
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Create SmartDataframe with PandasAI
        llm = get_pandasai_llm()
        smart_df = SmartDataframe(df, config={"llm": llm})
        
        # Get basic info about the dataset
        data_info = {
            "file_path": file_path,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records'),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        return {
            "status": "success",
            "data_info": data_info,
            "smart_dataframe": smart_df,
            "raw_dataframe": df
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading CSV: {str(e)}"
        }

def pandasai_query_tool(smart_dataframe: SmartDataframe, natural_language_query: str) -> Dict[str, Any]:
    """
    Executes natural language queries using PandasAI SmartDataframe.
    
    Args:
        smart_dataframe: PandasAI SmartDataframe object
        natural_language_query: User's natural language request
        
    Returns:
        Dictionary containing query results
    """
    try:
        # Execute the natural language query
        result = smart_dataframe.chat(natural_language_query)
        
        # Handle different types of results
        if isinstance(result, pd.DataFrame):
            return {
                "status": "success",
                "result_type": "dataframe",
                "data": result.to_dict('records'),
                "columns": list(result.columns),
                "shape": result.shape,
                "query": natural_language_query
            }
        elif isinstance(result, (int, float)):
            return {
                "status": "success",
                "result_type": "numeric",
                "value": result,
                "query": natural_language_query
            }
        elif isinstance(result, str):
            return {
                "status": "success",
                "result_type": "text",
                "text": result,
                "query": natural_language_query
            }
        else:
            return {
                "status": "success",
                "result_type": "other",
                "result": str(result),
                "query": natural_language_query
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error executing PandasAI query: {str(e)}",
            "query": natural_language_query
        }

def pandasai_analysis_tool(smart_dataframe: SmartDataframe, analysis_type: str = "descriptive") -> Dict[str, Any]:
    """
    Performs data analysis using PandasAI natural language capabilities.
    
    Args:
        smart_dataframe: PandasAI SmartDataframe object
        analysis_type: Type of analysis (descriptive, correlation, trend, etc.)
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        results = {"status": "success", "analysis_type": analysis_type, "insights": []}
        
        if analysis_type == "descriptive":
            # Get basic descriptive statistics
            desc_result = smart_dataframe.chat("Show descriptive statistics for all numeric columns")
            results["descriptive_stats"] = desc_result
            
            # Get missing values info
            missing_result = smart_dataframe.chat("Show missing values count for each column")
            results["missing_values"] = missing_result
            
            # Get data types
            dtypes_result = smart_dataframe.chat("Show data types for each column")
            results["data_types"] = dtypes_result
            
        elif analysis_type == "correlation":
            # Get correlation analysis
            corr_result = smart_dataframe.chat("Show correlation matrix for numeric columns")
            results["correlation"] = corr_result
            
            # Find strong correlations
            strong_corr_result = smart_dataframe.chat("Find pairs of columns with correlation > 0.7 or < -0.7")
            results["strong_correlations"] = strong_corr_result
            
        elif analysis_type == "trend":
            # Look for time-based patterns
            trend_result = smart_dataframe.chat("Identify any time-based trends in the data")
            results["trends"] = trend_result
            
        elif analysis_type == "outliers":
            # Detect outliers
            outlier_result = smart_dataframe.chat("Identify potential outliers in numeric columns")
            results["outliers"] = outlier_result
            
        elif analysis_type == "summary":
            # General data summary
            summary_result = smart_dataframe.chat("Provide a comprehensive summary of this dataset")
            results["summary"] = summary_result
            
        return results
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in PandasAI analysis: {str(e)}"
        }

def pandasai_visualization_tool(smart_dataframe: SmartDataframe, chart_request: str) -> Dict[str, Any]:
    """
    Generates visualizations using PandasAI natural language capabilities.
    
    Args:
        smart_dataframe: PandasAI SmartDataframe object
        chart_request: Natural language description of the desired visualization
        
    Returns:
        Dictionary containing visualization results
    """
    try:
        # Generate visualization using PandasAI
        result = smart_dataframe.chat(chart_request)
        
        return {
            "status": "success",
            "chart_request": chart_request,
            "result": result,
            "message": "Visualization generated successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating visualization: {str(e)}",
            "chart_request": chart_request
        }

def csv_data_manager_tool(file_paths: List[str]) -> Dict[str, Any]:
    """
    Manages multiple CSV files and creates SmartDataframes for each.
    Specifically designed for your hourly observation and daily flow data.
    
    Args:
        file_paths: List of paths to CSV files
        
    Returns:
        Dictionary containing loaded datasets and SmartDataframes
    """
    try:
        datasets = {}
        llm = get_pandasai_llm()
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            # Load CSV file
            df = pd.read_csv(file_path)
            smart_df = SmartDataframe(df, config={"llm": llm})
            
            # Determine dataset type based on filename
            filename = os.path.basename(file_path).lower()
            if 'hourly' in filename and 'obs' in filename:
                dataset_type = "hourly_observations"
            elif 'daily' in filename and ('flow' in filename or 'escape' in filename):
                dataset_type = "daily_flows"
            else:
                dataset_type = filename.replace('.csv', '').replace(' ', '_')
            
            datasets[dataset_type] = {
                "file_path": file_path,
                "dataframe": df,
                "smart_dataframe": smart_df,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict('records')
            }
        
        return {
            "status": "success",
            "datasets": datasets,
            "dataset_count": len(datasets),
            "available_types": list(datasets.keys())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error managing CSV data: {str(e)}"
        }

def pandasai_insight_tool(smart_dataframe: SmartDataframe, data_context: str = "") -> Dict[str, Any]:
    """
    Extracts business insights using PandasAI's natural language capabilities.
    
    Args:
        smart_dataframe: PandasAI SmartDataframe object
        data_context: Context about the data (e.g., "water flow data", "hourly observations")
        
    Returns:
        Dictionary containing extracted insights
    """
    try:
        insights = {
            "status": "success",
            "context": data_context,
            "key_insights": [],
            "recommendations": [],
            "data_quality_notes": []
        }
        
        # Get data quality insights
        quality_result = smart_dataframe.chat("Identify data quality issues such as missing values, outliers, or inconsistencies")
        insights["data_quality_notes"].append(str(quality_result))
        
        # Get key patterns and trends
        patterns_result = smart_dataframe.chat("What are the main patterns, trends, or anomalies in this data?")
        insights["key_insights"].append(str(patterns_result))
        
        # Get statistical insights
        stats_result = smart_dataframe.chat("What are the most important statistical findings from this data?")
        insights["key_insights"].append(str(stats_result))
        
        # Get recommendations based on data type
        if "hourly" in data_context.lower() or "observation" in data_context.lower():
            rec_result = smart_dataframe.chat("Based on this hourly observation data, what monitoring or operational recommendations can you provide?")
        elif "flow" in data_context.lower() or "daily" in data_context.lower():
            rec_result = smart_dataframe.chat("Based on this flow data, what water management recommendations can you provide?")
        else:
            rec_result = smart_dataframe.chat("What actionable recommendations can you provide based on this data?")
        
        insights["recommendations"].append(str(rec_result))
        
        return insights
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error extracting insights: {str(e)}"
        }

def visualization_generator_tool(data: List[Dict], chart_type: str = "auto") -> Dict[str, Any]:
    """
    Generates visualizations from data.
    
    Args:
        data: List of dictionaries containing the data
        chart_type: Type of chart to generate (auto, bar, line, scatter, histogram)
        
    Returns:
        Dictionary containing visualization information
    """
    try:
        if not data:
            return {
                "status": "error",
                "message": "No data provided for visualization"
            }
        
        df = pd.DataFrame(data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Analysis Visualizations', fontsize=16)
        
        visualizations_created = []
        
        # Chart 1: Distribution of first numeric column
        if numeric_columns:
            col = numeric_columns[0]
            axes[0, 0].hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue')
            axes[0, 0].set_title(f'Distribution of {col}')
            axes[0, 0].set_xlabel(col)
            axes[0, 0].set_ylabel('Frequency')
            visualizations_created.append(f"Histogram of {col}")
        
        # Chart 2: Bar chart if categorical data exists
        if categorical_columns and len(df) <= 50:  # Limit for readability
            col = categorical_columns[0]
            value_counts = df[col].value_counts().head(10)
            axes[0, 1].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[0, 1].set_title(f'Top 10 Values in {col}')
            axes[0, 1].set_xlabel(col)
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(range(len(value_counts)))
            axes[0, 1].set_xticklabels(value_counts.index, rotation=45, ha='right')
            visualizations_created.append(f"Bar chart of {col}")
        
        # Chart 3: Correlation heatmap if multiple numeric columns
        if len(numeric_columns) >= 2:
            correlation_matrix = df[numeric_columns].corr()
            im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title('Correlation Heatmap')
            axes[1, 0].set_xticks(range(len(correlation_matrix.columns)))
            axes[1, 0].set_yticks(range(len(correlation_matrix.columns)))
            axes[1, 0].set_xticklabels(correlation_matrix.columns, rotation=45)
            axes[1, 0].set_yticklabels(correlation_matrix.columns)
            visualizations_created.append("Correlation heatmap")
        
        # Chart 4: Scatter plot
        if len(numeric_columns) >= 2:
            axes[1, 1].scatter(df[numeric_columns[0]], df[numeric_columns[1]], alpha=0.6, color='green')
            axes[1, 1].set_title(f'{numeric_columns[0]} vs {numeric_columns[1]}')
            axes[1, 1].set_xlabel(numeric_columns[0])
            axes[1, 1].set_ylabel(numeric_columns[1])
            visualizations_created.append(f"Scatter plot: {numeric_columns[0]} vs {numeric_columns[1]}")
        
        # Remove empty subplots
        for i in range(2):
            for j in range(2):
                if not axes[i, j].has_data():
                    axes[i, j].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_analysis_charts_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "status": "success",
            "chart_file": filename,
            "visualizations_created": visualizations_created,
            "data_summary": {
                "total_rows": len(df),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating visualization: {str(e)}"
        }

def insight_extractor_tool(analysis_results: Dict, data_summary: Dict) -> Dict[str, Any]:
    """
    Extracts business insights from analysis results.
    
    Args:
        analysis_results: Results from statistical analysis
        data_summary: Summary of the analyzed data
        
    Returns:
        Dictionary containing extracted insights
    """
    try:
        insights = {
            "status": "success",
            "key_insights": [],
            "recommendations": [],
            "data_quality_notes": []
        }
        
        # Data quality insights
        if "missing_values" in analysis_results:
            missing_data = analysis_results["missing_values"]
            total_rows = analysis_results.get("total_rows", 0)
            
            for column, missing_count in missing_data.items():
                if missing_count > 0:
                    missing_percentage = (missing_count / total_rows) * 100
                    if missing_percentage > 10:
                        insights["data_quality_notes"].append(
                            f"Column '{column}' has {missing_percentage:.1f}% missing values"
                        )
        
        # Statistical insights
        if "descriptive_statistics" in analysis_results:
            desc_stats = analysis_results["descriptive_statistics"]
            
            for column, stats in desc_stats.items():
                mean_val = stats.get("mean", 0)
                std_val = stats.get("std", 0)
                
                if std_val > 0:
                    cv = (std_val / mean_val) * 100  # Coefficient of variation
                    if cv > 50:
                        insights["key_insights"].append(
                            f"'{column}' shows high variability (CV: {cv:.1f}%)"
                        )
                
                # Check for potential outliers
                q75 = stats.get("75%", 0)
                q25 = stats.get("25%", 0)
                iqr = q75 - q25
                max_val = stats.get("max", 0)
                
                if iqr > 0 and max_val > (q75 + 1.5 * iqr):
                    insights["key_insights"].append(
                        f"'{column}' may contain outliers (max value significantly above Q3)"
                    )
        
        # Correlation insights
        if "strong_correlations" in analysis_results:
            correlations = analysis_results["strong_correlations"]
            
            for corr in correlations:
                if corr["correlation"] > 0.8:
                    insights["key_insights"].append(
                        f"Strong positive correlation between '{corr['variable1']}' and '{corr['variable2']}' ({corr['correlation']:.2f})"
                    )
                elif corr["correlation"] < -0.8:
                    insights["key_insights"].append(
                        f"Strong negative correlation between '{corr['variable1']}' and '{corr['variable2']}' ({corr['correlation']:.2f})"
                    )
        
        # Trend insights
        if "trends" in analysis_results:
            trends = analysis_results["trends"]
            
            for column, trend_info in trends.items():
                direction = trend_info["direction"]
                slope = trend_info["slope"]
                
                if abs(slope) > 1:  # Significant trend
                    insights["key_insights"].append(
                        f"'{column}' shows a {direction} trend over time"
                    )
        
        # Generate recommendations
        if len(insights["data_quality_notes"]) > 0:
            insights["recommendations"].append(
                "Consider data cleaning for columns with high missing values"
            )
        
        if len(insights["key_insights"]) == 0:
            insights["key_insights"].append(
                "Data appears to be within normal ranges with no significant patterns detected"
            )
        
        insights["recommendations"].append(
            "Consider collecting additional data points for more robust analysis"
        )
        
        return insights
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error extracting insights: {str(e)}"
        }

# =============================================================================
# PHASE 3: AGENT IMPLEMENTATION
# =============================================================================

# 1. Data Loader Agent
data_loader = LlmAgent(
    name="data_loader",
    model="gemini-2.5-pro",
    tools=[load_csv_data_tool, csv_data_manager_tool],
    output_key="loaded_data",
    instruction="""You are a Data Loader Agent specializing in loading and preparing CSV data for analysis.

Your tasks:
1. Load CSV files and create PandasAI SmartDataframes
2. Identify the type of data (hourly observations, daily flows, etc.)
3. Provide initial data overview including:
   - Dataset shape and structure
   - Column information and data types
   - Missing values assessment
   - Sample data preview

For the specific CSV files:
- hourly_obs_Eijsden+Venlo+Megen_2016_2023.csv: Hourly observation data
- daily_escape_flows_and_gross_diversions_corrected_cleaned.csv: Daily flow data

Always validate data quality and report any issues found.""",
    description="Loads CSV data and creates PandasAI SmartDataframes for analysis"
)

# 2. Query Executor Agent
query_executor = LlmAgent(
    name="query_executor", 
    model="gemini-2.5-pro",
    tools=[pandasai_query_tool],
    output_key="query_results",
    instruction="""You are a Query Executor Agent specialized in executing natural language queries on data using PandasAI.

Your tasks:
1. Execute natural language queries on SmartDataframes
2. Handle different types of query results (DataFrames, numbers, text)
3. Provide context about the results
4. Suggest follow-up queries when appropriate

Types of queries you can handle:
- Statistical summaries ("Show average temperature by month")
- Filtering ("Show data where flow > 100")
- Aggregations ("Count observations per location") 
- Comparisons ("Compare flows between years")
- Trends ("Show trend over time")

Always provide clear explanations of the results and their significance.""",
    description="Executes natural language queries using PandasAI SmartDataframes"
)

# 3. Analysis Engine Agent
analysis_engine = LlmAgent(
    name="analysis_engine",
    model="gemini-2.5-pro", 
    tools=[pandasai_analysis_tool, pandasai_visualization_tool],
    output_key="analysis_results",
    instruction="""You are an Analysis Engine Agent powered by PandasAI for sophisticated data analysis.

Your tasks:
1. Perform statistical analysis using pandasai_analysis_tool
2. Generate visualizations using pandasai_visualization_tool
3. Analyze patterns in hourly observation and daily flow data
4. Identify anomalies, trends, and correlations

For the specific datasets:
- Hourly observations: Focus on temporal patterns, seasonal variations, measurement quality
- Daily flows: Analyze flow patterns, diversions, escape flows, operational efficiency

Analysis types you perform:
- Descriptive statistics and data quality assessment
- Correlation analysis between variables
- Trend analysis over time periods
- Outlier detection and anomaly identification
- Comparative analysis between locations or time periods

Always provide context-aware analysis considering the environmental/hydrological nature of the data.""",
    description="Performs data analysis and visualization using PandasAI capabilities"
)

# 4. Report Generation Agent
report_generator = LlmAgent(
    name="report_generator",
    model="gemini-2.5-pro",
    tools=[pandasai_insight_tool],
    output_key="final_report",
    instruction="""You are a Report Generation Agent that creates comprehensive reports for water management and environmental monitoring.

Your tasks:
1. Extract insights using pandasai_insight_tool
2. Generate reports tailored to environmental/hydrological data
3. Provide operational recommendations for water management
4. Highlight data quality and monitoring insights

Report structure for environmental data:
- Executive Summary (key findings and implications)
- Data Overview (monitoring period, locations, measurements)
- Key Findings (patterns, trends, anomalies)
- Data Quality Assessment (completeness, reliability, gaps)
- Environmental Insights (seasonal patterns, flow characteristics)
- Operational Recommendations (monitoring improvements, management actions)
- Technical Notes (methodology, limitations)

Focus areas:
- Hourly observations: Measurement quality, temporal patterns, equipment performance
- Daily flows: Flow management efficiency, diversion patterns, operational optimization

Writing style:
- Clear technical communication for water management professionals
- Quantified findings with statistical context
- Actionable recommendations for operations and monitoring
- Environmental and regulatory compliance considerations""",
    description="Generates comprehensive reports for environmental and water management data"
)

# 5. Root Orchestrator Agent
data_analysis_coordinator = LlmAgent(
    name="data_analysis_coordinator",
    model="gemini-2.5-pro",
    sub_agents=[data_loader, query_executor, analysis_engine, report_generator],
    instruction="""You are the Data Analysis Coordinator for Environmental and Water Management Data Analysis.

Your role:
1. Coordinate analysis of CSV files (hourly observations and daily flows)
2. Manage workflow from data loading to final insights
3. Provide specialized support for environmental data analysis
4. Guide users through complex hydrological data exploration

Workflow management:
- Route data loading tasks to data_loader
- Route natural language queries to query_executor
- Route analysis tasks to analysis_engine
- Route report generation to report_generator

Available datasets:
- hourly_obs_Eijsden+Venlo+Megen_2016_2023.csv: Hourly observation data
- daily_escape_flows_and_gross_diversions_corrected_cleaned.csv: Daily flow data

User interaction guidelines:
- Help users explore environmental and hydrological patterns
- Suggest relevant analysis based on data type
- Explain environmental significance of findings
- Provide context for water management applications
- Guide users through seasonal, temporal, and spatial analysis

Specializations:
- Water flow analysis and management
- Environmental monitoring data quality
- Temporal pattern recognition in environmental data
- Operational efficiency assessment for water systems

Always provide environmentally-informed insights and practical recommendations.""",
    description="Main coordinator for environmental data analysis using PandasAI and ADK"
)

# =============================================================================
# WORKFLOW AGENTS FOR PHASE 3
# =============================================================================

# Sequential Pipeline for Complete Data Analysis
complete_analysis_pipeline = SequentialAgent(
    name="complete_analysis_pipeline",
    sub_agents=[
        data_loader,         # Step 1: Load CSV data
        query_executor,      # Step 2: Execute queries
        analysis_engine,     # Step 3: Perform analysis
        report_generator     # Step 4: Generate report
    ],
    description="Complete sequential pipeline for environmental data analysis from CSV loading to final report"
)

# Parallel Agent for Multi-CSV Analysis
multi_csv_analyzer = ParallelAgent(
    name="multi_csv_analyzer",
    sub_agents=[
        LlmAgent(
            name="hourly_data_agent",
            model="gemini-2.5-pro", 
            tools=[pandasai_query_tool, pandasai_analysis_tool],
            instruction="Analyze hourly observation data focusing on temporal patterns and data quality",
            description="Specialized agent for hourly observation data analysis"
        ),
        LlmAgent(
            name="daily_flow_agent",
            model="gemini-2.5-pro",
            tools=[pandasai_query_tool, pandasai_analysis_tool], 
            instruction="Analyze daily flow data focusing on flow patterns and operational efficiency",
            description="Specialized agent for daily flow data analysis"
        )
    ],
    description="Analyzes multiple CSV datasets in parallel for comprehensive environmental insights"
)

# =============================================================================
# RUNNER AND SESSION SETUP
# =============================================================================

class PandasAIDataAnalysisSystem:
    """
    Main class for the PandasAI-powered Data Analysis Agent System.
    Provides easy interface for analyzing CSV files using natural language.
    """
    
    def __init__(self, csv_file_paths: List[str] = None):
        self.session_service = InMemorySessionService()
        self.app_name = "pandasai_data_analysis_system"
        self.csv_files = csv_file_paths or [
            "hourly_obs_Eijsden+Venlo+Megen_2016_2023.csv",
            "daily_escape_flows_and_gross_diversions_corrected_cleaned.csv"
        ]
        self.datasets = {}
        
        # Initialize runners for different workflows
        self.main_runner = Runner(
            agent=data_analysis_coordinator,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        self.pipeline_runner = Runner(
            agent=complete_analysis_pipeline,
            app_name=f"{self.app_name}_pipeline",
            session_service=self.session_service
        )
        
        self.parallel_runner = Runner(
            agent=multi_csv_analyzer,
            app_name=f"{self.app_name}_parallel",
            session_service=self.session_service
        )
    
    def create_session(self, user_id: str = "default_user") -> str:
        """Create a new session for a user."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id, 
            session_id=session_id
        )
        return session_id
    
    def load_csv_datasets(self) -> Dict[str, Any]:
        """
        Load CSV datasets using the data manager tool.
        
        Returns:
            Dictionary with loaded dataset information
        """
        try:
            result = csv_data_manager_tool(self.csv_files)
            if result["status"] == "success":
                self.datasets = result["datasets"]
                print(f"Successfully loaded {result['dataset_count']} datasets:")
                for dataset_type in result["available_types"]:
                    print(f"  - {dataset_type}: {self.datasets[dataset_type]['shape']}")
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def analyze_with_pandasai(self, query: str, dataset_type: str = "auto", user_id: str = "default_user", session_id: Optional[str] = None) -> str:
        """
        Analyze data using PandasAI with natural language query.
        
        Args:
            query: Natural language data analysis request
            dataset_type: Type of dataset to analyze ("hourly_observations", "daily_flows", or "auto")
            user_id: User identifier
            session_id: Session identifier (creates new if None)
            
        Returns:
            String containing the analysis results
        """
        if session_id is None:
            session_id = self.create_session(user_id)
        
        # Load datasets if not already loaded
        if not self.datasets:
            load_result = self.load_csv_datasets()
            if load_result["status"] != "success":
                return f"Error loading datasets: {load_result['message']}"
        
        # Auto-select dataset based on query if not specified
        if dataset_type == "auto":
            query_lower = query.lower()
            if any(word in query_lower for word in ["hourly", "hour", "observation", "obs"]):
                dataset_type = "hourly_observations"
            elif any(word in query_lower for word in ["daily", "flow", "escape", "diversion"]):
                dataset_type = "daily_flows"
            else:
                dataset_type = list(self.datasets.keys())[0]  # Default to first dataset
        
        if dataset_type not in self.datasets:
            return f"Dataset type '{dataset_type}' not found. Available: {list(self.datasets.keys())}"
        
        try:
            # Execute query directly using PandasAI
            smart_df = self.datasets[dataset_type]["smart_dataframe"]
            result = pandasai_query_tool(smart_df, query)
            
            if result["status"] == "success":
                return f"Query: {query}\nDataset: {dataset_type}\nResult: {result}"
            else:
                return f"Error: {result['message']}"
                
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def analyze_data(self, query: str, user_id: str = "default_user", session_id: Optional[str] = None) -> str:
        """
        Main method to analyze data using the full agent workflow.
        
        Args:
            query: Natural language data analysis request
            user_id: User identifier
            session_id: Session identifier (creates new if None)
            
        Returns:
            String containing the analysis results
        """
        if session_id is None:
            session_id = self.create_session(user_id)
        
        content = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
        
        print(f"Processing query: {query}")
        print("=" * 60)
        
        for event in self.main_runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response():
                return event.content.parts[0].text
        
        return "No response received from the analysis system."
    
    def run_pipeline_analysis(self, query: str, user_id: str = "default_user") -> str:
        """
        Run analysis using the sequential pipeline workflow.
        
        Args:
            query: Natural language data analysis request
            user_id: User identifier
            
        Returns:
            String containing the pipeline analysis results
        """
        session_id = self.create_session(user_id)
        
        content = types.Content(
            role="user", 
            parts=[types.Part(text=query)]
        )
        
        print(f"Running pipeline analysis for: {query}")
        print("=" * 60)
        
        for event in self.pipeline_runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response():
                return event.content.parts[0].text
        
        return "No response received from the pipeline system."

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def main():
    """
    Example usage of the PandasAI Data Analysis Agent System.
    """
    print("Initializing PandasAI Data Analysis Agent System...")
    
    # Initialize the system with your CSV files
    csv_files = [
        "hourly_obs_Eijsden+Venlo+Megen_2016_2023.csv",
        "daily_escape_flows_and_gross_diversions_corrected_cleaned.csv"
    ]
    
    analysis_system = PandasAIDataAnalysisSystem(csv_files)
    
    # Load the datasets
    print("Loading CSV datasets...")
    load_result = analysis_system.load_csv_datasets()
    
    if load_result["status"] == "success":
        print("✅ Datasets loaded successfully!")
    else:
        print(f"❌ Error loading datasets: {load_result['message']}")
        return
    
    # Example queries for your specific datasets
    example_queries = [
        # Hourly observation queries
        "Show average hourly values by location for the past month",
        "Identify any missing data patterns in the hourly observations",
        "What are the peak observation times during the day?",
        "Show seasonal patterns in the hourly data",
        "Detect any anomalies or outliers in the measurements",
        
        # Daily flow queries  
        "What is the trend in daily escape flows over time?",
        "Compare gross diversions between different periods",
        "Show the relationship between escape flows and diversions",
        "Identify periods with unusually high or low flows",
        "Calculate monthly averages for all flow measurements",
        
        # Cross-dataset analysis
        "Compare patterns between hourly and daily data",
        "Show correlation between observation frequency and flow patterns"
    ]
    
    print("\nPandasAI Data Analysis System ready!")
    print("\nExample queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\nUsage examples:")
    print("1. Direct PandasAI query:")
    print("   result = analysis_system.analyze_with_pandasai('Show average flows by month')")
    print("\n2. Full agent workflow:")
    print("   result = analysis_system.analyze_data('Analyze flow patterns and provide recommendations')")
    print("\n3. Pipeline analysis:")
    print("   result = analysis_system.run_pipeline_analysis('Generate a comprehensive water management report')")
    
    print("\nEnvironment setup required:")
    print("- Set GOOGLE_API_KEY or OPENAI_API_KEY environment variable")
    print("- Install pandasai: pip install pandasai")
    print("- Ensure CSV files are in the current directory")
    
    # Example of direct usage (commented out to avoid execution)
    # print("\n" + "="*60)
    # print("Example Analysis:")
    # result = analysis_system.analyze_with_pandasai("Show basic statistics for all numeric columns")
    # print(result)

if __name__ == "__main__":
    main()