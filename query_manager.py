"""Query history and analytics management"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class QueryManager:
    """Manage and analyze query history"""
    
    def __init__(self, history_file: str = "data/query_history.json"):
        self.history_file = history_file
        self.ensure_history_file()
    
    def ensure_history_file(self):
        """Create history file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            self.reset_history()

    def reset_history(self):
        """Reset query history to empty state"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([], f)
            logger.info("Query history has been reset")
        except Exception as e:
            logger.error(f"Error resetting query history: {str(e)}")
    
    def add_query(self, query: str, params: Dict[str, Any], success: bool):
        """Add a query to the history"""
        try:
            history = self.load_history()
            
            query_record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "parameters": params,
                "success": success
            }
            
            history.append(query_record)
            
            # Keep only last 1000 queries
            if len(history) > 1000:
                history = history[-1000:]
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error adding query to history: {str(e)}")
    
    def load_history(self) -> List[Dict]:
        """Load query history"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading query history: {str(e)}")
            return []
    
    def get_popular_queries(self, limit: int = 5) -> List[Dict]:
        """Get most frequently asked queries"""
        history = self.load_history()
        
        # Count query occurrences
        query_counts = {}
        for record in history:
            query = record["query"]
            query_counts[query] = query_counts.get(query, 0) + 1
        
        # Sort by frequency
        popular_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [{"query": q, "count": c} for q, c in popular_queries]
    
    def get_success_rate(self) -> float:
        """Calculate query success rate"""
        history = self.load_history()
        
        if not history:
            return 0.0
        
        successful = sum(1 for record in history if record["success"])
        return (successful / len(history)) * 100
    
    def get_parameter_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about parameter usage"""
        history = self.load_history()
        
        stats = {
            "states": {},
            "crops": {},
            "time_periods": {},
            "query_types": {}
        }
        
        for record in history:
            params = record.get("parameters", {})
            
            # Count states
            for state in params.get("states", []):
                stats["states"][state] = stats["states"].get(state, 0) + 1
            
            # Count crops
            for crop in params.get("crops", []):
                stats["crops"][crop] = stats["crops"].get(crop, 0) + 1
            
            # Count time periods
            time_period = params.get("time_period")
            if time_period:
                stats["time_periods"][str(time_period)] = \
                    stats["time_periods"].get(str(time_period), 0) + 1
            
            # Count query types
            query_type = params.get("query_type", "unknown")
            stats["query_types"][query_type] = \
                stats["query_types"].get(query_type, 0) + 1
        
        return stats

    def get_recent_queries(self, hours: int = 24) -> List[Dict]:
        """Get queries from the last N hours"""
        history = self.load_history()
        current_time = datetime.now()
        
        recent_queries = []
        for record in history:
            try:
                query_time = datetime.fromisoformat(record["timestamp"])
                hours_diff = (current_time - query_time).total_seconds() / 3600
                
                if hours_diff <= hours:
                    recent_queries.append(record)
            except Exception as e:
                logger.error(f"Error parsing query timestamp: {str(e)}")
                continue
        
        return recent_queries

    def get_geographic_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about geographic distribution of queries"""
        history = self.load_history()
        
        geo_stats = {}
        for record in history:
            params = record.get("parameters", {})
            states = params.get("states", [])
            
            for state in states:
                if state not in geo_stats:
                    geo_stats[state] = {
                        "query_count": 0,
                        "crops": {},
                        "success_rate": 0,
                        "last_query": None
                    }
                
                stats = geo_stats[state]
                stats["query_count"] += 1
                
                # Track crops
                for crop in params.get("crops", []):
                    stats["crops"][crop] = stats["crops"].get(crop, 0) + 1
                
                # Update success rate
                success = record.get("success", False)
                total = stats["query_count"]
                successful = stats.get("successful_queries", 0) + (1 if success else 0)
                stats["successful_queries"] = successful
                stats["success_rate"] = (successful / total) * 100  # Convert to percentage (0-100)
                
                # Update last query time
                try:
                    query_time = datetime.fromisoformat(record["timestamp"])
                    if not stats["last_query"] or query_time > stats["last_query"]:
                        stats["last_query"] = query_time
                except Exception as e:
                    logger.error(f"Error parsing query timestamp: {str(e)}")
        
        return geo_stats

    def get_today_stats(self) -> Dict[str, Any]:
        """Get statistics for queries made today"""
        history = self.load_history()
        current_time = datetime.now()
        
        today_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "hourly_distribution": [0] * 24,
            "top_states": {},
            "top_crops": {}
        }
        
        for record in history:
            try:
                query_time = datetime.fromisoformat(record["timestamp"])
                if query_time.date() == current_time.date():
                    today_stats["total_queries"] += 1
                    if record.get("success", False):
                        today_stats["successful_queries"] += 1
                    
                    # Update hourly distribution
                    today_stats["hourly_distribution"][query_time.hour] += 1
                    
                    # Update state and crop stats
                    params = record.get("parameters", {})
                    for state in params.get("states", []):
                        today_stats["top_states"][state] = \
                            today_stats["top_states"].get(state, 0) + 1
                    
                    for crop in params.get("crops", []):
                        today_stats["top_crops"][crop] = \
                            today_stats["top_crops"].get(crop, 0) + 1
            except Exception as e:
                logger.error(f"Error parsing query timestamp: {str(e)}")
                continue
        
        # Sort top states and crops
        today_stats["top_states"] = dict(sorted(
            today_stats["top_states"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        today_stats["top_crops"] = dict(sorted(
            today_stats["top_crops"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        return today_stats