import json
import requests
from typing import Dict, Any, List
from datetime import datetime
import logging

class MCPProtocol:
    """
    Model Context Protocol implementation for agent communication
    """
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.headers = {'Content-Type': 'application/json'}
    
    def send_to_architect(self, message: Dict, agent_type: str) -> Dict:
        """Send message to architect agent via MCP"""
        endpoint = f"{self.base_url}/api/mcp/architect"
        
        payload = {
            'message': message,
            'sender': agent_type,
            'timestamp': datetime.utcnow().isoformat(),
            'protocol': 'MCP'
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            return response.json()
        except Exception as e:
            logging.error(f"MCP communication failed: {e}")
            return {'error': str(e), 'status': 'failed'}