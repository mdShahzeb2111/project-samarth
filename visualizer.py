"""Data visualization utilities for agricultural and climate data analysis"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, Tuple
import pandas as pd
from io import BytesIO
import base64

class DataVisualizer:
    """Generate various visualizations for data analysis"""
    
    @staticmethod
    def create_figure(width: int = 12, height: int = 6) -> Tuple[plt.Figure, plt.Axes]:
        """Create a new figure with given dimensions"""
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111)
        return fig, ax
    
    @staticmethod
    def save_plot_to_base64(fig: plt.Figure) -> str:
        """Convert a matplotlib figure to base64 string"""
        img_bytes = BytesIO()
        fig.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        plt.close(fig)
        return base64.b64encode(img_bytes.read()).decode()

    def plot_rainfall_comparison(self, data: pd.DataFrame, states: list, years: int) -> str:
        """Create rainfall comparison visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(data=data, x='state', y='rainfall', ax=ax1)
        ax1.set_title('Rainfall Distribution by State')
        ax1.set_ylabel('Rainfall (mm)')
        
        # Time series
        pivot_data = data.pivot(index='year', columns='state', values='rainfall')
        pivot_data.plot(ax=ax2, marker='o')
        ax2.set_title('Rainfall Trends')
        ax2.set_ylabel('Rainfall (mm)')
        
        return self.save_plot_to_base64(fig)

    def plot_crop_production(self, data: pd.DataFrame, crop: str, state: str) -> str:
        """Create crop production visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Production trend
        data.plot(x='year', y='production', ax=ax1, marker='o')
        ax1.set_title(f'{crop} Production Trend in {state}')
        ax1.set_ylabel('Production (tonnes)')
        
        # Yield distribution
        sns.boxplot(data=data, y='yield', ax=ax2)
        ax2.set_title('Yield Distribution')
        
        # Area vs Production scatter
        sns.scatterplot(data=data, x='area', y='production', ax=ax3)
        ax3.set_title('Area vs Production')
        
        # Monthly distribution if available
        if 'month' in data.columns:
            monthly_avg = data.groupby('month')['production'].mean()
            monthly_avg.plot(kind='bar', ax=ax4)
            ax4.set_title('Average Monthly Production')
        else:
            ax4.remove()
        
        plt.tight_layout()
        return self.save_plot_to_base64(fig)

    def plot_climate_impact(self, data: pd.DataFrame, climate_var: str, crop: str) -> str:
        """Create climate impact visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot with regression line
        sns.regplot(data=data, x=climate_var, y='production', ax=ax1)
        ax1.set_title(f'{climate_var.title()} vs Production')
        
        # Time series of both variables
        ax2.plot(data['year'], data[climate_var], label=climate_var.title())
        ax2.plot(data['year'], data['production'], label='Production')
        ax2.set_title('Trends Over Time')
        ax2.legend()
        
        # Monthly patterns if available
        if 'month' in data.columns:
            monthly_data = data.groupby('month')[[climate_var, 'production']].mean()
            monthly_data.plot(ax=ax3)
            ax3.set_title('Monthly Patterns')
        
        # Correlation heatmap
        corr_matrix = data[[climate_var, 'production', 'yield', 'area']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title('Correlation Matrix')
        
        plt.tight_layout()
        return self.save_plot_to_base64(fig)

    def plot_regional_summary(self, data: pd.DataFrame, metric: str) -> str:
        """Create regional summary visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Regional distribution
        sns.boxplot(data=data, x='region', y=metric, ax=ax1)
        ax1.set_title(f'{metric.title()} by Region')
        plt.xticks(rotation=45)
        
        # Time series by region
        pivot_data = data.pivot_table(
            index='year',
            columns='region',
            values=metric,
            aggfunc='mean'
        )
        pivot_data.plot(ax=ax2, marker='o')
        ax2.set_title(f'{metric.title()} Trends by Region')
        
        # Top states
        top_states = data.groupby('state')[metric].mean().nlargest(10)
        top_states.plot(kind='bar', ax=ax3)
        ax3.set_title(f'Top 10 States by {metric.title()}')
        plt.xticks(rotation=45)
        
        # Seasonal patterns if available
        if 'month' in data.columns:
            seasonal = data.pivot_table(
                index='month',
                columns='region',
                values=metric,
                aggfunc='mean'
            )
            seasonal.plot(ax=ax4, marker='o')
            ax4.set_title('Seasonal Patterns by Region')
        else:
            ax4.remove()
        
        plt.tight_layout()
        return self.save_plot_to_base64(fig)