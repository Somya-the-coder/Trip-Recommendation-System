import pandas as pd
from datetime import datetime, timedelta
import os
import json
import requests
from pytrends.request import TrendReq
import time
import random

class TrendingEngine:
    def __init__(self, cache_file="static/data/trending_cache.json"):
        """Initialize the trending engine with cache file location."""
        self.cache_file = cache_file
        self.trending_cache = None
        self.cache_timestamp = None
        
        # Load cache if it exists
        self._load_cache()
        
        # Sample destinations as fallback
        self.sample_destinations = [
            {"id": 1, "name": "Bali, Indonesia", "description": "Experience the beautiful beaches, vibrant culture, and stunning landscapes of Bali.", "image": "bali.jpg", "trend_score": 95},
            {"id": 2, "name": "Kyoto, Japan", "description": "Discover ancient temples, traditional gardens, and authentic Japanese experiences.", "image": "kyoto.jpg", "trend_score": 92},
            {"id": 3, "name": "Amalfi Coast, Italy", "description": "Enjoy breathtaking cliffside views, Mediterranean cuisine, and charming coastal towns.", "image": "amalfi.jpg", "trend_score": 89},
            {"id": 4, "name": "Santorini, Greece", "description": "Relax on the stunning white-washed island with its blue domes and breathtaking sunsets.", "image": "santorini.jpg", "trend_score": 87},
            {"id": 5, "name": "Marrakech, Morocco", "description": "Explore the vibrant markets, historic palaces, and rich cultural heritage of this ancient city.", "image": "marrakech.jpg", "trend_score": 84},
            {"id": 6, "name": "Cape Town, South Africa", "description": "Discover diverse landscapes from Table Mountain to stunning beaches and wildlife.", "image": "capetown.jpg", "trend_score": 82},
            {"id": 7, "name": "Reykjavik, Iceland", "description": "Experience the land of fire and ice with its volcanoes, geysers, and northern lights.", "image": "reykjavik.jpg", "trend_score": 81},
            {"id": 8, "name": "Tokyo, Japan", "description": "Experience the perfect blend of traditional culture and cutting-edge technology in Japan's capital.", "image": "tokyo.jpg", "trend_score": 80},
            {"id": 9, "name": "Barcelona, Spain", "description": "Enjoy Gaudi's architecture, Mediterranean beaches, and vibrant street life.", "image": "barcelona.jpg", "trend_score": 78},
            {"id": 10, "name": "New York City, USA", "description": "Discover the vibrant energy of the Big Apple with its iconic skyline and diverse neighborhoods.", "image": "nyc.jpg", "trend_score": 77}
        ]
    
    def _load_cache(self):
        """Load cached trending data if available."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.trending_cache = cache_data.get('destinations', [])
                    cache_time = cache_data.get('timestamp')
                    if cache_time:
                        self.cache_timestamp = datetime.fromisoformat(cache_time)
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save trending data to cache."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                'destinations': self.trending_cache,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_trending_from_google_trends(self):
        """Fetch trending destinations using Google Trends API (pytrends)."""
        try:
            # Initialize pytrends
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # Define travel-related keywords to check
            travel_keywords = [
                'travel to', 'visit', 'vacation in', 'flights to',
                'hotels in', 'tourism'
            ]
            
            trending_destinations = []
            
            # Get worldwide popular destinations as base
            potential_destinations = [
                'Paris', 'Bali', 'London', 'New York', 'Tokyo', 'Rome',
                'Barcelona', 'Dubai', 'Sydney', 'Santorini', 'Maldives',
                'Amsterdam', 'Singapore', 'Bangkok', 'Kyoto', 'Istanbul',
                'Prague', 'Marrakech', 'Reykjavik', 'Cape Town', 'Cancun',
                'Venice', 'Hong Kong', 'San Francisco', 'Phuket'
            ]
            
            # Check trends for each destination with travel keywords
            for destination in potential_destinations:
                kw_list = [f"{keyword} {destination}" for keyword in travel_keywords[:1]]  # Use first keyword to avoid rate limiting
                
                try:
                    # Get trend data for the past 90 days
                    pytrends.build_payload(kw_list, cat=67, timeframe='now 90-d')  # Cat 67 is travel category
                    interest_over_time = pytrends.interest_over_time()
                    
                    if not interest_over_time.empty:
                        # Calculate trend score based on average interest and recent growth
                        avg_interest = interest_over_time[kw_list[0]].mean()
                        
                        # Calculate growth by comparing last 30 days vs previous 30 days
                        recent = interest_over_time[kw_list[0]].iloc[-30:].mean()
                        previous = interest_over_time[kw_list[0]].iloc[-60:-30].mean()
                        growth = ((recent - previous) / previous) * 100 if previous > 0 else 0
                        
                        # Combine into a single trend score
                        trend_score = (0.7 * avg_interest) + (0.3 * growth) if growth > 0 else avg_interest
                        
                        trending_destinations.append({
                            "name": f"{destination}",
                            "description": f"A trending destination with growing interest based on recent travel searches.",
                            "trend_score": float(trend_score),
                            "growth": float(growth if growth > 0 else 0)
                        })
                
                except Exception as e:
                    print(f"Error processing {destination}: {e}")
                
                # Sleep to avoid hitting rate limits
                time.sleep(0.5)
            
            # Sort by trend score
            trending_destinations.sort(key=lambda x: x["trend_score"], reverse=True)
            
            # Add some context to the descriptions
            descriptions = {
                "Paris": "Explore iconic landmarks like the Eiffel Tower and world-class museums in the City of Light.",
                "Bali": "Experience the beautiful beaches, vibrant culture, and stunning landscapes of this Indonesian paradise.",
                "London": "Discover historic sites, diverse neighborhoods, and cutting-edge culture in England's capital.",
                "New York": "Experience the energy of the Big Apple with its iconic skyline and diverse neighborhoods.",
                "Tokyo": "Immerse yourself in Japan's blend of ultramodern and traditional culture in this dynamic metropolis.",
                "Rome": "Wander through ancient ruins, Vatican treasures, and enjoy authentic Italian cuisine.",
                "Barcelona": "Enjoy Gaudi's architecture, Mediterranean beaches, and vibrant street life.",
                "Dubai": "Marvel at futuristic architecture, luxury shopping, and desert adventures.",
                "Sydney": "Experience stunning harbor views, beautiful beaches, and Australia's laid-back lifestyle.",
                "Santorini": "Relax on this stunning Greek island with its iconic white buildings and breathtaking sunsets.",
                "Maldives": "Escape to pristine beaches, overwater bungalows, and world-class snorkeling.",
                "Amsterdam": "Explore picturesque canals, historic buildings, and world-renowned museums.",
                "Singapore": "Discover a futuristic city-state with amazing food, gardens, and multicultural experiences.",
                "Bangkok": "Immerse yourself in Thailand's vibrant street life, temples, and legendary cuisine.",
                "Kyoto": "Experience Japan's ancient temples, traditional gardens, and authentic cultural heritage.",
                "Istanbul": "Bridge Europe and Asia with Byzantine wonders, Ottoman palaces, and vibrant bazaars.",
                "Prague": "Wander through fairy-tale architecture and cobblestone streets in the heart of Europe.",
                "Marrakech": "Explore colorful markets, stunning palaces, and the gateway to Morocco's desert adventures.",
                "Reykjavik": "Experience the land of fire and ice with Iceland's volcanoes, geysers, and northern lights.",
                "Cape Town": "Discover diverse landscapes from Table Mountain to stunning beaches and wildlife.",
                "Cancun": "Enjoy pristine Caribbean beaches, ancient Mayan ruins, and vibrant nightlife.",
                "Venice": "Navigate the romantic canals and historic architecture of this unique Italian city.",
                "Hong Kong": "Experience the perfect blend of East and West with stunning skylines and incredible food.",
                "San Francisco": "Explore iconic landmarks, diverse neighborhoods, and innovative culture by the bay.",
                "Phuket": "Relax on Thailand's largest island with beautiful beaches and vibrant nightlife."
            }
            
            # Update descriptions
            for dest in trending_destinations:
                city_name = dest["name"].split(",")[0] if "," in dest["name"] else dest["name"]
                if city_name in descriptions:
                    dest["description"] = descriptions[city_name]
            
            return trending_destinations
        
        except Exception as e:
            print(f"Error fetching from Google Trends: {e}")
            return []
    
    def get_trending_from_travel_api(self):
        """Fetch trending destinations using travel API (Skyscanner/Amadeus)."""
        try:
            # You would need to sign up for API keys for these services
            # This is a placeholder implementation showing how you would integrate

            # Option 1: Skyscanner API
            # API_KEY = "your_skyscanner_api_key"
            # url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/browsequotes/v1.0/US/USD/en-US/anywhere/anywhere/anytime"
            # headers = {
            #     'x-rapidapi-key': API_KEY,
            #     'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com"
            # }
            # response = requests.get(url, headers=headers)
            # data = response.json()
            
            # Option 2: Amadeus API
            # from amadeus import Client, ResponseError
            # amadeus = Client(
            #     client_id='YOUR_API_KEY',
            #     client_secret='YOUR_API_SECRET'
            # )
            # try:
            #     response = amadeus.shopping.flight_destinations.get(
            #         origin='PAR',
            #         maxPrice=200
            #     )
            #     data = response.data
            # except ResponseError as error:
            #     print(error)
            
            # Since we don't have actual API keys, return an empty list
            # In a real implementation, you would process the API response here
            return []
            
        except Exception as e:
            print(f"Error fetching from Travel API: {e}")
            return []
    
    def get_trending_destinations(self, limit=6, force_refresh=False):
        """Get the current trending destinations.
        
        Args:
            limit: Number of trending destinations to return
            force_refresh: Whether to force a refresh of the trending data
        
        Returns:
            List of trending destination objects
        """
        # Check if we need to refresh the cache
        current_time = datetime.now()
        cache_expired = (
            self.cache_timestamp is None or
            current_time - self.cache_timestamp > timedelta(hours=6)
        )
        
        if self.trending_cache is None or force_refresh or cache_expired:
            # Try to get data from Google Trends
            trending_google = self.get_trending_from_google_trends()
            
            # Try to get data from Travel API
            trending_travel_api = self.get_trending_from_travel_api()
            
            # Combine and deduplicate results
            combined_trending = trending_google + trending_travel_api
            
            # If we got data from APIs, use it
            if combined_trending:
                # Sort by trend score
                combined_trending.sort(key=lambda x: x.get("trend_score", 0), reverse=True)
                self.trending_cache = combined_trending
            else:
                # Fall back to sample data with some randomization
                sample_copy = self.sample_destinations.copy()
                
                # Add some randomness to trend scores to simulate changes
                for dest in sample_copy:
                    dest["trend_score"] += random.uniform(-5, 5)
                
                # Sort by trending score
                sample_copy.sort(key=lambda x: x["trend_score"], reverse=True)
                self.trending_cache = sample_copy
            
            self.cache_timestamp = current_time
            self._save_cache()
        
        # Return the requested number of trending destinations
        return self.trending_cache[:limit]
    
    def update_trending_data(self):
        """Update the trending data - could be called by a scheduler."""
        trending = self.get_trending_destinations(force_refresh=True)
        return trending

# For testing
if __name__ == "__main__":
    trending_engine = TrendingEngine()
    trending_destinations = trending_engine.get_trending_destinations()
    print(trending_destinations)