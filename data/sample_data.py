import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SampleDataProvider:
    """Provides sample data for development and testing"""
    
    def __init__(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate base date range
        self.years = range(2015, 2024)
        self.states = [
            'Maharashtra', 'Karnataka', 'Gujarat', 'Punjab', 
            'Uttar Pradesh', 'Madhya Pradesh', 'Bihar'
        ]
        self.crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']
        
    def get_crop_production_data(self) -> pd.DataFrame:
        """Generate sample crop production data"""
        data = []
        
        for year in self.years:
            for state in self.states:
                for crop in self.crops:
                    # Generate realistic production values
                    base_production = np.random.uniform(100000, 500000)
                    # Add yearly trend
                    trend = (year - 2015) * 1000
                    # Add some randomness
                    variation = np.random.normal(0, 10000)
                    
                    production = base_production + trend + variation
                    
                    # Generate area and yield
                    area = np.random.uniform(50000, 200000)
                    yield_value = production / area
                    
                    data.append({
                        'year': year,
                        'state': state,
                        'crop': crop,
                        'production': round(production, 2),
                        'area': round(area, 2),
                        'yield': round(yield_value, 2)
                    })
        
        df = pd.DataFrame(data)
        df.attrs['data_type'] = 'agriculture'
        return df
    
    def get_land_use_data(self) -> pd.DataFrame:
        """Generate sample land use data"""
        data = []
        land_use_types = ['Agricultural', 'Forest', 'Urban', 'Wasteland', 'Water Bodies']
        
        for year in self.years:
            for state in self.states:
                total_area = 1000000  # Total area in hectares
                # Generate random percentages that sum to 100
                percentages = np.random.dirichlet(np.ones(len(land_use_types))) * 100
                
                for land_use, percentage in zip(land_use_types, percentages):
                    area = (total_area * percentage) / 100
                    data.append({
                        'year': year,
                        'state': state,
                        'land_use_type': land_use,
                        'area': round(area, 2),
                        'percentage': round(percentage, 2)
                    })
        
        df = pd.DataFrame(data)
        df.attrs['data_type'] = 'land_use'
        return df
    
    def get_rainfall_data(self) -> pd.DataFrame:
        """Generate sample rainfall data"""
        data = []
        
        for year in self.years:
            for state in self.states:
                # Generate monthly rainfall
                for month in range(1, 13):
                    # Base rainfall varies by season
                    if month in [6, 7, 8, 9]:  # Monsoon
                        base_rainfall = np.random.uniform(200, 400)
                    elif month in [3, 4, 5]:  # Summer
                        base_rainfall = np.random.uniform(20, 80)
                    else:  # Winter
                        base_rainfall = np.random.uniform(10, 50)
                    
                    # Add variation
                    rainfall = base_rainfall + np.random.normal(0, 20)
                    rainfall = max(0, rainfall)  # Ensure non-negative
                    
                    data.append({
                        'year': year,
                        'month': month,
                        'state': state,
                        'rainfall': round(rainfall, 2)
                    })
        
        df = pd.DataFrame(data)
        df.attrs['data_type'] = 'rainfall'
        return df
    
    def get_temperature_data(self) -> pd.DataFrame:
        """Generate sample temperature data"""
        data = []
        
        for year in self.years:
            for state in self.states:
                for month in range(1, 13):
                    # Base temperature varies by season
                    if month in [3, 4, 5]:  # Summer
                        base_temp = np.random.uniform(35, 45)
                    elif month in [12, 1, 2]:  # Winter
                        base_temp = np.random.uniform(10, 20)
                    else:  # Spring/Autumn
                        base_temp = np.random.uniform(25, 35)
                    
                    # Daily temperature range
                    temp_range = np.random.uniform(8, 15)
                    max_temp = base_temp + (temp_range/2)
                    min_temp = base_temp - (temp_range/2)
                    
                    data.append({
                        'year': year,
                        'month': month,
                        'state': state,
                        'max_temperature': round(max_temp, 2),
                        'min_temperature': round(min_temp, 2),
                        'avg_temperature': round(base_temp, 2)
                    })
        
        df = pd.DataFrame(data)
        df.attrs['data_type'] = 'temperature'
        return df