from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
import logging
from typing import Dict, Any
from datetime import datetime

# Import after type definitions to avoid circular imports
mcp_bp = Blueprint('mcp', __name__)

@mcp_bp.route('/api/mcp/architect', methods=['POST'])
@login_required
def architect_endpoint():
    """MCP endpoint for architect agent"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        sender = data.get('sender', 'unknown')
        message = data.get('message', {})
        
        logging.info(f"📡 MCP Message from {sender}: {message.get('action', 'unknown')}")
        
        # Process based on message type
        action = message.get('action')
        
        if action == 'data_collection_complete':
            response = handle_data_collection_complete(message)
        elif action == 'chart_generation_complete':
            response = handle_chart_generation_complete(message)
        elif action == 'insight_generation_complete':
            response = handle_insight_generation_complete(message)
        else:
            response = create_response(False, error="Unknown action")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"MCP architect error: {e}")
        return jsonify(create_response(False, error=str(e))), 500

@mcp_bp.route('/api/mcp/data_collector', methods=['POST'])
@login_required
def data_collector_endpoint():
    """MCP endpoint for data collector agent"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        message = data.get('message', {})
        
        # Import here to avoid circular imports
        from agents.data_collector import DataCollectorAgent
        agent = DataCollectorAgent()
        
        if message.get('type') == 'data_request':
            user_data = agent.collect_user_data(current_user.id)
            response = create_response(True, user_data)
        else:
            response = create_response(False, error="Unknown message type")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(create_response(False, error=str(e))), 500

def handle_data_collection_complete(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle data collection completion"""
    user_id = message.get('user_id')
    transaction_count = message.get('transaction_count', 0)
    
    logging.info(f"✅ Data collection complete for user {user_id}: {transaction_count} transactions")
    
    # Trigger chart generation
    try:
        from agents.architect_agent import ArchitectAgent
        architect = current_app.architect_agent
        architect.trigger_chart_generation(user_id)
    except Exception as e:
        logging.error(f"Failed to trigger chart generation: {e}")
    
    return create_response(True, {'next_action': 'chart_generation_triggered'})

def handle_chart_generation_complete(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle chart generation completion"""
    user_id = message.get('user_id')
    chart_count = message.get('chart_count', 0)
    
    logging.info(f"✅ Chart generation complete for user {user_id}: {chart_count} charts")
    
    # Trigger insight generation
    try:
        from agents.architect_agent import ArchitectAgent
        architect = current_app.architect_agent
        architect.trigger_insight_generation(user_id)
    except Exception as e:
        logging.error(f"Failed to trigger insight generation: {e}")
    
    return create_response(True, {'next_action': 'insight_generation_triggered'})

def handle_insight_generation_complete(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle insight generation completion"""
    user_id = message.get('user_id')
    insight_count = message.get('insight_count', 0)
    
    logging.info(f"✅ Insight generation complete for user {user_id}: {insight_count} insights")
    
    return create_response(True, {'workflow_complete': True})

def create_response(success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """Create standardized response"""
    from datetime import datetime
    return {
        'success': success,
        'data': data,
        'error': error,
        'timestamp': datetime.utcnow().isoformat()
    }

# Add to app/routes/mcp_api.py
@mcp_bp.route('/test-mcp')
@login_required
def test_mcp():
    try:
        from protocols.mcp_protocol import MCPProtocol
        mcp = MCPProtocol()
        
        test_message = {
            'action': 'test_communication',
            'user_id': current_user.id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Test sending message
        response = mcp.send_to_architect(test_message, 'tester')
        
        return jsonify({
            'mcp_working': True,
            'response': response,
            'message_sent': test_message
        })
    except Exception as e:
        return jsonify({
            'mcp_working': False,
            'error': str(e)
        })