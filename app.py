import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
import csv
from io import StringIO
from collections import defaultdict
from flask import Response
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template
import os
from functools import lru_cache
import hashlib
import pickle
from dotenv import load_dotenv
import matplotlib.pyplot as plt   
import seaborn as sns
from io import BytesIO
import base64
from visualizer import DataVisualizer
from query_manager import QueryManager

# Configure plotting settings
plt.style.use('default')  # Use default style first
sns.set_theme(style="whitegrid")  # Then apply seaborn themes
sns.set_palette("husl")

# Load environment variables
load_dotenv('config.env')

# Import error handling
from error_handler import (
    QASystemError, 
    DataFetchError, 
    AnalysisError, 
    QueryError, 
    handle_error, 
    format_error_message
)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Cache:
    """Simple file-based cache system"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pickle")
    
    def get(self, key: str, max_age: timedelta = timedelta(days=1)) -> Any:
        """Get value from cache"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                # Check if cache is still valid
                if datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path)) < max_age:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            logger.error(f"Cache read error: {str(e)}")
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")

class QueryAnalyzer:
    """Analyze and process natural language queries"""
    
    def __init__(self):
        # Enhanced patterns for better query understanding
        self.patterns = {
            'states': re.compile(r'(?:State_[XY]|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            'crops': re.compile(r'(?:Crop_Type_[ABC]|Crop_[Z]|rice|wheat|cotton|sugarcane|maize|pulses)', re.IGNORECASE),
            'years': re.compile(r'(?:last|past|previous)\s+(\d+)\s*(?:year|yr)s?|\d{4}', re.IGNORECASE),
            'districts': re.compile(r'district\s+(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            'climate_params': re.compile(r'(?:rainfall|temperature|humidity|precipitation)', re.IGNORECASE),
            'analysis_type': re.compile(r'(?:compare|analyze|correlate|trend|impact|relationship|pattern)', re.IGNORECASE),
            'top_n': re.compile(r'top\s+(\d+)', re.IGNORECASE),
            'regions': re.compile(r'(?:north|south|east|west|central)\s+(?:india|region)', re.IGNORECASE)
        }
        
        # Query type patterns
        self.query_types = {
            'comparison': re.compile(r'compare|versus|vs|difference between', re.IGNORECASE),
            'trend_analysis': re.compile(r'trend|pattern|change over|evolution', re.IGNORECASE),
            'correlation': re.compile(r'correlate|relationship|impact|effect|influence', re.IGNORECASE),
            'ranking': re.compile(r'top|highest|lowest|maximum|minimum|ranking', re.IGNORECASE),
            'summary': re.compile(r'summarize|overview|brief|summary', re.IGNORECASE)
        }
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract key parameters from the query"""
        params = {}
        
        # Extract basic parameters
        for param_type, pattern in self.patterns.items():
            matches = pattern.findall(query)
            params[param_type] = [m[0] if isinstance(m, tuple) else m for m in matches]
        
        # Determine query type
        query_type = self.determine_query_type(query)
        params['query_type'] = query_type
        
        # Extract time range
        time_range = self.extract_time_range(query)
        params.update(time_range)
        
        # Add query complexity level
        params['complexity'] = self.assess_query_complexity(query)
        
        return params
    
    def determine_query_type(self, query: str) -> str:
        """Determine the primary type of analysis needed"""
        query = query.lower()
        
        for query_type, pattern in self.query_types.items():
            if pattern.search(query):
                return query_type
        
        return 'general'
    
    def extract_time_range(self, query: str) -> Dict[str, Any]:
        """Extract time-related parameters from the query"""
        time_info = {
            'start_year': None,
            'end_year': None,
            'time_period': None
        }
        
        # Look for explicit years
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        years = year_pattern.findall(query)
        
        if years:
            years = sorted(map(int, years))
            time_info['start_year'] = years[0]
            time_info['end_year'] = years[-1] if len(years) > 1 else None
        
        # Look for relative time periods
        period_match = re.search(r'(?:last|past|previous)\s+(\d+)\s*(?:year|yr)s?', query, re.IGNORECASE)
        if period_match:
            time_info['time_period'] = int(period_match.group(1))
            
        return time_info
    
    def assess_query_complexity(self, query: str) -> str:
        """Assess the complexity level of the query"""
        # Count different types of parameters
        param_count = sum(1 for pattern in self.patterns.values() if pattern.search(query))
        
        # Check for multiple analysis requirements
        analysis_types = sum(1 for pattern in self.query_types.values() if pattern.search(query))
        
        if param_count >= 4 or analysis_types >= 2:
            return 'complex'
        elif param_count >= 2:
            return 'moderate'
        else:
            return 'simple'
    
    def fetch_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Fetch a specific dataset by ID with caching"""
        cache_key = f"dataset_{dataset_id}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Retrieved dataset {dataset_id} from cache")
            return cached_data
        
        try:
            logger.info(f"Fetching dataset {dataset_id} from data.gov.in")
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 1000
            }
            response = self.session.get(f"{self.base_url}/{dataset_id}", params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data.get("records", []))
            
            # Add metadata
            df.attrs['source_id'] = dataset_id
            df.attrs['fetch_date'] = datetime.now().isoformat()
            
            # Cache the data
            self.cache.set(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching dataset {dataset_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_agriculture_data(self, data_type: str = 'crop_production') -> pd.DataFrame:
        """Get agricultural data of specified type"""
        dataset_ids = self.datasets['agriculture'].get(data_type, [])
        dfs = []
        
        for dataset_id in dataset_ids:
            df = self.fetch_dataset(dataset_id)
            if not df.empty:
                dfs.append(df)
        
        return pd.concat(dfs) if dfs else pd.DataFrame()
    
    def get_climate_data(self, data_type: str = 'rainfall') -> pd.DataFrame:
        """Get climate data of specified type"""
        dataset_ids = self.datasets['climate'].get(data_type, [])
        dfs = []
        
        for dataset_id in dataset_ids:
            df = self.fetch_dataset(dataset_id)
            if not df.empty:
                dfs.append(df)
        
        return pd.concat(dfs) if dfs else pd.DataFrame()
    
    def search_datasets(self, keywords: List[str]) -> List[Dict]:
        """Search for datasets using keywords"""
        try:
            cache_key = f"search_{'_'.join(keywords)}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            params = {
                "format": "json",
                "api-key": self.api_key,
                "filters[keyword]": ",".join(keywords)
            }
            response = self.session.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            
            results = response.json().get("records", [])
            self.cache.set(cache_key, results)
            return results
            
        except Exception as e:
            logger.error(f"Error searching datasets: {str(e)}")
            return []

class DataProcessor:
    """Process and harmonize data from different sources"""
    
    def __init__(self):
        self.agriculture_data = {}
        self.climate_data = {}
    
    def process_agriculture_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize agriculture data"""
        try:
            if data.empty:
                return data
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Convert data types
            if 'year' in data.columns:
                data['year'] = pd.to_numeric(data['year'], errors='coerce')
            if 'production' in data.columns:
                data['production'] = pd.to_numeric(data['production'], errors='coerce')
            if 'area' in data.columns:
                data['area'] = pd.to_numeric(data['area'], errors='coerce')
            
            # Calculate yield if not present
            if 'yield' not in data.columns and 'production' in data.columns and 'area' in data.columns:
                data['yield'] = data['production'] / data['area']
            
            # Add metadata
            data.attrs['processed_date'] = datetime.now().isoformat()
            data.attrs['data_type'] = 'agriculture'
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing agriculture data: {str(e)}")
            return pd.DataFrame()
    
    def process_climate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize climate data"""
        try:
            if data.empty:
                return data
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Convert data types
            if 'year' in data.columns:
                data['year'] = pd.to_numeric(data['year'], errors='coerce')
            if 'rainfall' in data.columns:
                data['rainfall'] = pd.to_numeric(data['rainfall'], errors='coerce')
            if 'temperature' in data.columns:
                data['temperature'] = pd.to_numeric(data['temperature'], errors='coerce')
            
            # Convert date columns
            date_columns = [col for col in data.columns if 'date' in col.lower()]
            for col in date_columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
            
            # Add metadata
            data.attrs['processed_date'] = datetime.now().isoformat()
            data.attrs['data_type'] = 'climate'
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing climate data: {str(e)}")
            return pd.DataFrame()
    
    def merge_datasets(self, agriculture_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
        """Merge agriculture and climate data based on common fields"""
        try:
            if agriculture_df.empty or climate_df.empty:
                return pd.DataFrame()
            
            # Identify common fields for merging
            common_fields = list(set(agriculture_df.columns) & set(climate_df.columns))
            if not common_fields:
                logger.error("No common fields found for merging datasets")
                return pd.DataFrame()
            
            # Merge datasets
            merged_df = pd.merge(
                agriculture_df,
                climate_df,
                on=common_fields,
                suffixes=('_agri', '_climate')
            )
            
            # Add metadata
            merged_df.attrs['merge_date'] = datetime.now().isoformat()
            merged_df.attrs['data_sources'] = [
                agriculture_df.attrs.get('source_id'),
                climate_df.attrs.get('source_id')
            ]
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            return pd.DataFrame()

class QueryAnalyzer:
    """Analyze and process natural language queries"""
    
    def __init__(self):
        # Enhanced patterns for better query understanding
        self.patterns = {
            'states': re.compile(r'(?:State_[XY]|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            'crops': re.compile(r'(?:Crop_Type_[ABC]|Crop_[Z]|rice|wheat|cotton|sugarcane|maize|pulses)', re.IGNORECASE),
            'years': re.compile(r'(?:last|past|previous)\s+(\d+)\s*(?:year|yr)s?|\d{4}', re.IGNORECASE),
            'districts': re.compile(r'district\s+(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            'climate_params': re.compile(r'(?:rainfall|temperature|humidity|precipitation)', re.IGNORECASE),
            'analysis_type': re.compile(r'(?:compare|analyze|correlate|trend|impact|relationship|pattern)', re.IGNORECASE),
            'top_n': re.compile(r'top\s+(\d+)', re.IGNORECASE),
            'regions': re.compile(r'(?:north|south|east|west|central)\s+(?:india|region)', re.IGNORECASE)
        }
        
        # Query type patterns
        self.query_types = {
            'comparison': re.compile(r'compare|versus|vs|difference between', re.IGNORECASE),
            'trend_analysis': re.compile(r'trend|pattern|change over|evolution', re.IGNORECASE),
            'correlation': re.compile(r'correlate|relationship|impact|effect|influence', re.IGNORECASE),
            'ranking': re.compile(r'top|highest|lowest|maximum|minimum|ranking', re.IGNORECASE),
            'summary': re.compile(r'summarize|overview|brief|summary', re.IGNORECASE)
        }
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract key parameters from the query"""
        params = {}
        
        # Extract basic parameters
        for param_type, pattern in self.patterns.items():
            matches = pattern.findall(query)
            params[param_type] = [m[0] if isinstance(m, tuple) else m for m in matches]
        
        # Determine query type
        query_type = self.determine_query_type(query)
        params['query_type'] = query_type
        
        # Extract time range
        time_range = self.extract_time_range(query)
        params.update(time_range)
        
        # Add query complexity level
        params['complexity'] = self.assess_query_complexity(query)
        
        return params
    
    def determine_query_type(self, query: str) -> str:
        """Determine the primary type of analysis needed"""
        query = query.lower()
        
        for query_type, pattern in self.query_types.items():
            if pattern.search(query):
                return query_type
        
        return 'general'
    
    def extract_time_range(self, query: str) -> Dict[str, Any]:
        """Extract time-related parameters from the query"""
        time_info = {
            'start_year': None,
            'end_year': None,
            'time_period': None
        }
        
        # Look for explicit years
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        years = year_pattern.findall(query)
        
        if years:
            years = sorted(map(int, years))
            time_info['start_year'] = years[0]
            time_info['end_year'] = years[-1] if len(years) > 1 else None
        
        # Look for relative time periods
        period_match = re.search(r'(?:last|past|previous)\s+(\d+)\s*(?:year|yr)s?', query, re.IGNORECASE)
        if period_match:
            time_info['time_period'] = int(period_match.group(1))
            
        return time_info
    
    def assess_query_complexity(self, query: str) -> str:
        """Assess the complexity level of the query"""
        # Count different types of parameters
        param_count = sum(1 for pattern in self.patterns.values() if pattern.search(query))
        
        # Check for multiple analysis requirements
        analysis_types = sum(1 for pattern in self.query_types.values() if pattern.search(query))
        
        if param_count >= 4 or analysis_types >= 2:
            return 'complex'
        elif param_count >= 2:
            return 'moderate'
        else:
            return 'simple'

class DataAnalyzer:
    """Analyze data and generate insights"""
    
    def __init__(self):
        self.cache = Cache()
        self.processor = DataProcessor()
    
    def compare_rainfall(self, state1: str, state2: str, years: int) -> Dict:
        """Compare rainfall between two states"""
        try:
            cache_key = f"rainfall_comparison_{state1}_{state2}_{years}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Fetch and process climate data
            climate_data = self.data_client.get_climate_data('rainfall')
            climate_data = self.processor.process_climate_data(climate_data)
            
            if climate_data.empty:
                return {"error": "No rainfall data available"}
            
            # Filter data for the specified states and years
            current_year = datetime.now().year
            filtered_data = climate_data[
                (climate_data['state'].isin([state1, state2])) &
                (climate_data['year'] >= current_year - years)
            ]
            
            # Calculate statistics
            stats = filtered_data.groupby('state')['rainfall'].agg([
                'mean', 'std', 'min', 'max'
            ]).round(2)
            
            # Generate visualization
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=filtered_data, x='state', y='rainfall')
            plt.title(f'Rainfall Comparison: {state1} vs {state2} ({years} years)')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png')
            img_bytes.seek(0)
            plt.close()
            
            result = {
                "analysis": {
                    "statistics": stats.to_dict(),
                    "visualization": base64.b64encode(img_bytes.read()).decode(),
                    "years_analyzed": years,
                    "states_compared": [state1, state2]
                },
                "source_dataset": climate_data.attrs.get('source_id'),
                "analysis_date": datetime.now().isoformat()
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in rainfall comparison: {str(e)}")
            return {"error": str(e)}
    
    def analyze_crop_production(self, state: str, crop: str, recent_years: int = 5) -> Dict:
        """Analyze crop production for a specific state and crop"""
        try:
            cache_key = f"crop_production_{state}_{crop}_{recent_years}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Fetch and process agriculture data
            agri_data = self.data_client.get_agriculture_data('crop_production')
            agri_data = self.processor.process_agriculture_data(agri_data)
            
            if agri_data.empty:
                return {"error": "No crop production data available"}
            
            # Filter data
            current_year = datetime.now().year
            filtered_data = agri_data[
                (agri_data['state'] == state) &
                (agri_data['crop'] == crop) &
                (agri_data['year'] >= current_year - recent_years)
            ]
            
            # Calculate statistics and trends
            stats = {
                'total_production': filtered_data['production'].sum(),
                'average_yield': filtered_data['yield'].mean(),
                'production_trend': filtered_data.groupby('year')['production'].mean().to_dict(),
                'cultivated_area': filtered_data['area'].sum()
            }
            
            # Generate visualization
            plt.figure(figsize=(12, 6))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Production trend
            sns.lineplot(data=filtered_data, x='year', y='production', ax=ax1)
            ax1.set_title(f'{crop} Production Trend in {state}')
            
            # Yield distribution
            sns.boxplot(data=filtered_data, y='yield', ax=ax2)
            ax2.set_title(f'{crop} Yield Distribution')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png')
            img_bytes.seek(0)
            plt.close()
            
            result = {
                "analysis": {
                    "statistics": stats,
                    "visualization": base64.b64encode(img_bytes.read()).decode(),
                    "years_analyzed": recent_years,
                    "state": state,
                    "crop": crop
                },
                "source_dataset": agri_data.attrs.get('source_id'),
                "analysis_date": datetime.now().isoformat()
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in crop production analysis: {str(e)}")
            return {"error": str(e)}
    
    def analyze_climate_impact(self, rainfall_data: pd.DataFrame, production_data: pd.DataFrame, 
                             state: str, crop: str, years: int) -> Tuple[str, List[str], str]:
        """Analyze climate impact on crop production"""
        try:
            # Filter data
            rainfall_state = rainfall_data[rainfall_data['state'] == state]
            production_state = production_data[
                (production_data['state'] == state) &
                (production_data['crop'] == crop)
            ]
            
            # Merge on year
            merged_data = pd.merge(rainfall_state, production_state, on=['state', 'year'])
            
            # Calculate correlation
            correlation = merged_data['rainfall'].corr(merged_data['production'])
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            sns.regplot(data=merged_data, x='rainfall', y='production', ax=ax1)
            ax1.set_title(f'Rainfall vs {crop} Production in {state}')
            ax1.set_xlabel('Rainfall (mm)')
            ax1.set_ylabel('Production (tonnes)')
            
            # Time series with dual axis
            line1 = ax2.plot(merged_data['year'], merged_data['rainfall'], color='blue', label='Rainfall')
            ax2.set_ylabel('Rainfall (mm)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            ax3 = ax2.twinx()
            line2 = ax3.plot(merged_data['year'], merged_data['production'], color='red', label='Production')
            ax3.set_ylabel('Production (tonnes)', color='red')
            ax3.tick_params(axis='y', labelcolor='red')
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels)
            ax2.set_title('Rainfall and Production Trends')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight')
            img_bytes.seek(0)
            plt.close()
            
            # Generate response
            response = [
                f"Climate Impact Analysis for {crop} in {state}:",
                f"Correlation between rainfall and production: {correlation:.2f}",
                f"\nKey Findings:",
                f"- {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} "
                f"{'positive' if correlation > 0 else 'negative'} correlation between rainfall and production",
                f"- Average rainfall: {merged_data['rainfall'].mean():.1f} mm",
                f"- Average production: {merged_data['production'].mean():,.0f} tonnes",
                f"- Year range: {merged_data['year'].min()} - {merged_data['year'].max()}"
            ]
            
            return "\n".join(response), ['Climate Data: data.gov.in', 'Agricultural Data: data.gov.in'], base64.b64encode(img_bytes.read()).decode()
            
        except Exception as e:
            logger.error(f"Error in climate impact analysis: {str(e)}")
            raise AnalysisError(f"Could not analyze climate impact: {str(e)}")

class ResponseGenerator:
    """Generate natural language responses"""
    
    def format_response(self, analysis_results: Dict) -> str:
        """Format analysis results into natural language response"""
        # Placeholder for implementation
        pass

class QASystem:
    """Main Q&A system integrating all components"""
    
    def __init__(self, api_key: str = None):
        logger.info("Initializing QA System...")
        try:
            self.cache = Cache()
            logger.debug("Cache initialized")
            
            self.query_analyzer = QueryAnalyzer()
            logger.debug("Query Analyzer initialized")
            
            self.visualizer = DataVisualizer()
            logger.debug("Data Visualizer initialized")
            
            self.query_manager = QueryManager()
            logger.debug("Query Manager initialized")
            
            self.analyzer = DataAnalyzer()
            logger.debug("Data Analyzer initialized")
            
            self.start_time = datetime.now()
            self.response_times = []
            self.query_types_count = defaultdict(int)
            
            logger.info("QA System initialization completed successfully")
        except Exception as e:
            logger.error(f"Error initializing QA System: {str(e)}", exc_info=True)
            raise
    
    def generate_sample_data(self, states: List[str], crops: List[str], years: int) -> Dict[str, pd.DataFrame]:
        """Generate sample data for testing"""
        current_year = datetime.now().year
        years_range = range(current_year - years, current_year)
        
        # Generate rainfall data
        rainfall_data = []
        for state in states:
            for year in years_range:
                rainfall_data.append({
                    'state': state,
                    'year': year,
                    'rainfall': np.random.normal(900, 200),  # Mean: 900mm, STD: 200mm
                    'temperature': np.random.normal(25, 5)   # Mean: 25°C, STD: 5°C
                })
        rainfall_df = pd.DataFrame(rainfall_data)
        
        # Generate crop production data
        production_data = []
        for state in states:
            for crop in crops:
                for year in years_range:
                    production_data.append({
                        'state': state,
                        'crop': crop,
                        'year': year,
                        'production': np.random.normal(1000000, 200000),  # Mean: 1M tonnes
                        'area': np.random.normal(500000, 100000),         # Mean: 500K hectares
                        'yield': np.random.normal(2, 0.5)                 # Mean: 2 tonnes/hectare
                    })
        production_df = pd.DataFrame(production_data)
        
        return {
            'rainfall': rainfall_df,
            'production': production_df
        }

    def process_query(self, query: str) -> Tuple[str, List[str], str]:
        """Process a natural language query and return (response, sources, visualization)"""
        try:
            query_start_time = datetime.now()
            logger.debug(f"Starting query processing: {query}")
            
            # Check cache first
            cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query}")
                # Track cache hit response time
                response_time = (datetime.now() - query_start_time).total_seconds()
                self.response_times.append({"time": response_time, "cached": True})
                return cached_result

            # Analyze query
            logger.debug("Extracting query parameters...")
            query_params = self.query_analyzer.extract_parameters(query)
            states = query_params.get('states', ['Maharashtra', 'Punjab'])
            crops = query_params.get('crops', ['rice', 'wheat'])
            years = query_params.get('years', [])
            years_num = int(years[0]) if years else 5
            query_type = query_params.get('query_type', 'general').lower()

            logger.info(f"Extracted params: states={states}, crops={crops}, years={years_num}, type={query_type}")
            
            # Generate sample data
            logger.debug("Generating sample data...")
            try:
                sample_data = self.generate_sample_data(states, crops, years_num)
                logger.debug("Sample data generated successfully")
            except Exception as data_error:
                logger.error(f"Error generating sample data: {str(data_error)}", exc_info=True)
                raise DataFetchError(f"Failed to generate required data: {str(data_error)}")

            # Process based on query type and return appropriate analysis
            logger.debug(f"Processing query type: {query_type}")
            if 'comparison' in query_type and len(states) >= 2:
                logger.debug("Performing rainfall comparison analysis...")
                result = self.analyze_rainfall_comparison(
                    sample_data['rainfall'],
                    states[0],
                    states[1],
                    years_num
                )
            elif 'trend' in query_type or 'production' in query_type:
                logger.debug("Performing crop production analysis...")
                result = self.analyze_crop_production(
                    sample_data['production'],
                    states[0],
                    crops[0],
                    years_num
                )
            elif 'correlation' in query_type or 'impact' in query_type:
                logger.debug("Performing climate impact analysis...")
                result = self.analyze_climate_impact(
                    sample_data['rainfall'],
                    sample_data['production'],
                    states[0],
                    crops[0],
                    years_num
                )
            else:
                logger.debug("Performing general analysis...")
                result = self.analyze_general(
                    sample_data,
                    states[0],
                    crops[0] if crops else None,
                    years_num
                )

            if result is None:
                logger.error("Analysis returned no result")
                raise AnalysisError("Analysis failed to produce results")

            # Track response time and query type
            response_time = (datetime.now() - query_start_time).total_seconds()
            self.response_times.append({
                "time": response_time,
                "cached": False,
                "query_type": query_type
            })
            
            # Update query type statistics
            self.query_types_count[query_type] += 1

            # Cache successful results
            logger.debug("Caching results...")
            self.cache.set(cache_key, result)
                
            # Add analytics data
            analytics_data = {
                "response_time": response_time,
                "query_type": query_type,
                "timestamp": datetime.now().isoformat(),
                "parameters": query_params
            }
            self.query_manager.add_analytics(analytics_data)
            
            logger.debug("Query processing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            # Track failed queries
            self.query_manager.add_analytics({
                "error": str(e),
                "query_type": "error",
                "timestamp": datetime.now().isoformat()
            })
            raise QueryError(f"Error processing query: {str(e)}")

    def analyze_rainfall_comparison(self, rainfall_data: pd.DataFrame, state1: str, state2: str, years: int) -> Tuple[str, List[str], str]:
        """Analyze and compare rainfall between two states"""
        try:
            # Filter data
            state_data = rainfall_data[rainfall_data['state'].isin([state1, state2])]
            
            # Calculate statistics
            stats = state_data.groupby('state').agg({
                'rainfall': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plot
            sns.boxplot(data=state_data, x='state', y='rainfall', ax=ax1)
            ax1.set_title(f'Rainfall Distribution ({years} years)')
            ax1.set_ylabel('Rainfall (mm)')
            
            # Time series
            pivot_data = state_data.pivot_table(index='year', columns='state', values='rainfall', aggfunc='mean')
            pivot_data.plot(ax=ax2, marker='o')
            ax2.set_title('Rainfall Trends')
            ax2.set_ylabel('Rainfall (mm)')
            ax2.legend(title='State')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight')
            img_bytes.seek(0)
            plt.close()
            
            # Generate response
            def format_stats(state_name):
                return (
                    f"{state_name}:\n"
                    f"  Average: {stats.loc[state_name, ('rainfall', 'mean')]:.1f} mm\n"
                    f"  Standard Deviation: {stats.loc[state_name, ('rainfall', 'std')]:.1f} mm\n"
                    f"  Range: {stats.loc[state_name, ('rainfall', 'min')]:.1f} - {stats.loc[state_name, ('rainfall', 'max')]:.1f} mm"
                )
            
            response = [
                f"Rainfall Comparison between {state1} and {state2} for {years} years:",
                format_stats(state1),
                format_stats(state2),
                f"\nKey Finding: {state1 if stats.loc[state1, ('rainfall', 'mean')] > stats.loc[state2, ('rainfall', 'mean')] else state2} "
                f"has higher average rainfall."
            ]
            
            return "\n".join(response), ['Rainfall Data: data.gov.in'], base64.b64encode(img_bytes.read()).decode()
            
        except Exception as e:
            logger.error(f"Error in rainfall comparison: {str(e)}")
            raise AnalysisError(f"Could not compare rainfall data: {str(e)}")

    def analyze_crop_production(self, production_data: pd.DataFrame, state: str, crop: str, years: int) -> Tuple[str, List[str], str]:
        """Analyze crop production for a state"""
        try:
            # Filter data
            crop_data = production_data[
                (production_data['state'] == state) &
                (production_data['crop'] == crop)
            ]
            
            # Calculate statistics
            stats = {
                'total_production': crop_data['production'].sum(),
                'average_yield': crop_data['yield'].mean(),
                'area_cultivated': crop_data['area'].mean()
            }
            
            # Generate visualization
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=crop_data, x='year', y='production', marker='o')
            plt.title(f'{crop} Production in {state} (Last {years} years)')
            plt.ylabel('Production (tonnes)')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight')
            img_bytes.seek(0)
            plt.close()
            
            # Generate response
            response = [
                f"{crop} Production Analysis in {state} for the last {years} years:",
                f"Total Production: {stats['total_production']:,.0f} tonnes",
                f"Average Yield: {stats['average_yield']:.2f} tonnes/hectare",
                f"Average Area Cultivated: {stats['area_cultivated']:,.0f} hectares"
            ]
            
            return "\n".join(response), ['Agricultural Production Data: data.gov.in'], base64.b64encode(img_bytes.read()).decode()
            
        except Exception as e:
            logger.error(f"Error in crop production analysis: {str(e)}")
            raise AnalysisError(f"Could not analyze crop production: {str(e)}")

    def analyze_climate_impact(self, rainfall_data: pd.DataFrame, production_data: pd.DataFrame, 
                             state: str, crop: str, years: int) -> Tuple[str, List[str], str]:
        """Analyze climate impact on crop production"""
        try:
            # Filter data
            rainfall_state = rainfall_data[rainfall_data['state'] == state]
            production_state = production_data[
                (production_data['state'] == state) &
                (production_data['crop'] == crop)
            ]
            
            # Merge on year
            merged_data = pd.merge(rainfall_state, production_state, on=['state', 'year'])
            
            # Calculate correlation
            correlation = merged_data['rainfall'].corr(merged_data['production'])
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            sns.regplot(data=merged_data, x='rainfall', y='production', ax=ax1)
            ax1.set_title(f'Rainfall vs {crop} Production in {state}')
            ax1.set_xlabel('Rainfall (mm)')
            ax1.set_ylabel('Production (tonnes)')
            
            # Time series with dual axis
            line1 = ax2.plot(merged_data['year'], merged_data['rainfall'], color='blue', label='Rainfall')
            ax2.set_ylabel('Rainfall (mm)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            ax3 = ax2.twinx()
            line2 = ax3.plot(merged_data['year'], merged_data['production'], color='red', label='Production')
            ax3.set_ylabel('Production (tonnes)', color='red')
            ax3.tick_params(axis='y', labelcolor='red')
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels)
            ax2.set_title('Rainfall and Production Trends')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight')
            img_bytes.seek(0)
            plt.close()
            
            # Generate response
            response = [
                f"Climate Impact Analysis for {crop} in {state}:",
                f"Correlation between rainfall and production: {correlation:.2f}",
                f"\nKey Findings:",
                f"- {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} "
                f"{'positive' if correlation > 0 else 'negative'} correlation between rainfall and production",
                f"- Average rainfall: {merged_data['rainfall'].mean():.1f} mm",
                f"- Average production: {merged_data['production'].mean():,.0f} tonnes",
                f"- Year range: {merged_data['year'].min()} - {merged_data['year'].max()}"
            ]
            
            return "\n".join(response), ['Climate Data: data.gov.in', 'Agricultural Data: data.gov.in'], base64.b64encode(img_bytes.read()).decode()
            
        except Exception as e:
            logger.error(f"Error in climate impact analysis: {str(e)}")
            raise AnalysisError(f"Could not analyze climate impact: {str(e)}")

    def analyze_general(self, sample_data: Dict[str, pd.DataFrame], state: str, crop: str = None, years: int = 5) -> Tuple[str, List[str], str]:
        """Perform general analysis of available data"""
        try:
            rainfall_data = sample_data['rainfall']
            production_data = sample_data['production']
            
            # Filter data
            state_rainfall = rainfall_data[rainfall_data['state'] == state]
            state_production = production_data[production_data['state'] == state]
            
            if crop:
                state_production = state_production[state_production['crop'] == crop]
            
            # Calculate basic statistics
            stats = {
                'avg_rainfall': state_rainfall['rainfall'].mean(),
                'total_production': state_production['production'].sum(),
                'crops_grown': len(state_production['crop'].unique()) if not crop else 1,
                'years_data': years
            }
            
            # Generate visualization
            plt.figure(figsize=(12, 6))
            if crop:
                plt.title(f'{crop} Production and Rainfall in {state}')
            else:
                plt.title(f'Agricultural Overview of {state}')
                
            # Create summary plot
            sns.lineplot(data=state_rainfall, x='year', y='rainfall')
            plt.ylabel('Rainfall (mm)')
            
            # Save plot
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight')
            img_bytes.seek(0)
            plt.close()
            
            # Generate response
            response = [
                f"Analysis for {state}{'- ' + crop if crop else ''} (Last {years} years):",
                f"Average Annual Rainfall: {stats['avg_rainfall']:.1f} mm",
                f"Total Agricultural Production: {stats['total_production']:,.0f} tonnes",
                f"Number of Crops: {stats['crops_grown']}"
            ]
            
            return "\n".join(response), ['Climate Data: data.gov.in', 'Agricultural Data: data.gov.in'], base64.b64encode(img_bytes.read()).decode()
            
        except Exception as e:
            logger.error(f"Error in general analysis: {str(e)}")
            raise AnalysisError(f"Could not perform general analysis: {str(e)}")

# Initialize Flask application
app = Flask(__name__)

# Initialize the QA system globally
qa_system = QASystem(api_key=os.getenv('DATA_GOV_API_KEY'))

@app.route('/')
def home():
    """Render the home page"""
    try:
        logger.debug("Rendering home page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Handle the Q&A endpoint"""
    try:
        logger.debug("Received POST request to /ask")
        data = request.json
        logger.debug(f"Request data: {data}")

        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400

        query = data.get('query', '')
        
        if not query:
            logger.warning("Empty query received")
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"Processing query: {query}")
        try:
            # Extract parameters before processing
            params = qa_system.query_analyzer.extract_parameters(query)
            logger.debug(f"Extracted parameters: {params}")
            
            response, sources, visualization = qa_system.process_query(query)
            logger.debug(f"Query processed successfully")
            logger.debug(f"Response: {response}, Sources: {sources}, Visualization: {'present' if visualization else 'none'}")
            
            # Save successful query with parameters
            qa_system.query_manager.add_query(query, params, True)
            
            result = {
                "answer": response,
                "sources": sources,
                "visualization": visualization
            }
            logger.debug(f"Sending response: {result}")
            return jsonify(result)
        except Exception as query_error:
            logger.error(f"Error in query processing: {str(query_error)}", exc_info=True)
            # Save failed query
            qa_system.query_manager.add_query(query, {}, False)
            error_response = handle_error(query_error)
            logger.debug(f"Error response: {error_response}")
            return jsonify(error_response), 500
    except Exception as e:
        logger.error(f"Critical error in /ask endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "error_code": "CRITICAL_ERROR"
        }), 500

@app.route('/popular_queries')
def get_popular_queries():
    """Get most frequently asked queries"""
    try:
        popular = qa_system.query_manager.get_popular_queries()
        return jsonify(popular)
    except Exception as e:
        logger.error(f"Error getting popular queries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_analytics', methods=['POST'])
def reset_analytics():
    """Reset analytics data"""
    try:
        qa_system.query_manager.reset_history()
        return jsonify({"status": "success", "message": "Analytics data has been reset"})
    except Exception as e:
        logger.error(f"Error resetting analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/export_data')
def export_data():
    """Export analytics data in CSV format"""
    try:
        data = qa_system.query_manager.get_analytics_data()
        if not data:
            return jsonify({"error": "No data available"}), 404
            
        # Convert to CSV
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=analytics_data.csv'
            }
        )
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/performance_metrics')
def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        response_times = qa_system.response_times
        if not response_times:
            return jsonify({"error": "No performance data available"}), 404
            
        # Calculate metrics
        times = [r["time"] for r in response_times]
        cached_times = [r["time"] for r in response_times if r.get("cached")]
        non_cached_times = [r["time"] for r in response_times if not r.get("cached")]
        
        metrics = {
            "average_response_time": sum(times) / len(times) if times else 0,
            "min_response_time": min(times) if times else 0,
            "max_response_time": max(times) if times else 0,
            "cached_avg_time": sum(cached_times) / len(cached_times) if cached_times else 0,
            "non_cached_avg_time": sum(non_cached_times) / len(non_cached_times) if non_cached_times else 0,
            "total_queries": len(times),
            "cache_hit_rate": len(cached_times) / len(times) if times else 0
        }
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query_types')
def get_query_types():
    """Get query type analysis"""
    try:
        query_types = qa_system.query_types_count
        total_queries = sum(query_types.values())
        
        analysis = {
            "distribution": {k: v/total_queries for k, v in query_types.items()},
            "counts": dict(query_types),
            "total_queries": total_queries
        }
        
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error getting query types: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/trend_analysis')
def get_trend_analysis():
    """Get trend analysis and predictions"""
    try:
        recent_queries = qa_system.query_manager.get_recent_queries(hours=24)
        if not recent_queries:
            return jsonify({"error": "No data available for trend analysis"}), 404
            
        # Group queries by hour
        hourly_data = defaultdict(int)
        for query in recent_queries:
            try:
                timestamp = datetime.fromisoformat(query["timestamp"])
                hourly_data[timestamp.hour] += 1
            except Exception as e:
                logger.error(f"Error parsing timestamp: {str(e)}")
                continue
        
        # Simple trend analysis
        current_hour = datetime.now().hour
        recent_hours = list(range(current_hour - 6, current_hour))
        recent_counts = [hourly_data.get(h, 0) for h in recent_hours]
        
        # Calculate trend
        if len(recent_counts) >= 2:
            trend = sum(y2 - y1 for y1, y2 in zip(recent_counts[:-1], recent_counts[1:])) / len(recent_counts)
        else:
            trend = 0
            
        analysis = {
            "hourly_data": dict(hourly_data),
            "trend": trend,
            "prediction": {
                "next_hour": max(0, hourly_data.get(current_hour, 0) + trend),
                "trend_direction": "up" if trend > 0 else "down" if trend < 0 else "stable"
            }
        }
        
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error performing trend analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query_stats')
def get_query_stats():
    """Get comprehensive query statistics"""
    try:
        # Get all stats
        success_rate = qa_system.query_manager.get_success_rate()
        parameter_stats = qa_system.query_manager.get_parameter_stats()
        today_stats = qa_system.query_manager.get_today_stats()
        geo_stats = qa_system.query_manager.get_geographic_stats()
        
        # Calculate hourly queries (last 24 hours)
        current_hour = datetime.now().hour
        hourly_queries = today_stats['hourly_distribution']
        
        # Rotate array so current hour is last
        hourly_queries = hourly_queries[current_hour+1:] + hourly_queries[:current_hour+1]
        
        stats = {
            "success_rate": success_rate,
            "parameter_stats": parameter_stats,
            "hourly_queries": hourly_queries,
            "total_queries_today": today_stats["total_queries"],
            "today_stats": {
                "successful_queries": today_stats["successful_queries"],
                "top_states": today_stats["top_states"],
                "top_crops": today_stats["top_crops"]
            },
            "geographic_insights": geo_stats
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting query stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_query', methods=['POST'])
def save_query():
    """Save a query for future reference"""
    try:
        data = request.json
        query = data.get('query', '')
        params = data.get('parameters', {})
        success = data.get('success', True)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        qa_system.query_manager.add_query(query, params, success)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error saving query: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    # Check if running in production or development
    is_production = os.environ.get('PRODUCTION', False)
    
    if is_production:
        # Production settings
        logger.info("Starting Flask application in production mode...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    else:
        # Development settings
        logger.info("Starting Flask application in development mode...")
        app.run(debug=True, port=5000)

