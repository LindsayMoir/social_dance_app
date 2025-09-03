"""
conversation_manager.py

This module provides the ConversationManager class for handling multi-turn chatbot conversations.
It manages session creation, context tracking, and conversation history storage using PostgreSQL.

Features:
- Anonymous session-based conversations
- Context preservation across multiple queries
- Intent classification for query refinement
- Entity extraction from user messages
- Conversation cleanup for inactive sessions

Classes:
    ConversationManager: Main class for conversation management
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any


class ConversationManager:
    def __init__(self, db_handler):
        """
        Initialize ConversationManager with database handler.
        
        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db_handler = db_handler
        
    def create_or_get_conversation(self, session_token: str) -> str:
        """
        Create a new conversation or retrieve existing one by session token.
        
        Args:
            session_token: Unique session identifier from frontend
            
        Returns:
            str: conversation_id (UUID)
        """
        # Check if conversation exists
        query = "SELECT conversation_id FROM conversations WHERE session_token = :session_token AND is_active = true"
        result = self.db_handler.execute_query(query, {"session_token": session_token})
        
        if result:
            conversation_id = str(result[0][0])
            # Update last_activity
            self._update_last_activity(conversation_id)
            return conversation_id
        
        # Create new conversation
        insert_query = """
            INSERT INTO conversations (session_token, created_at, last_activity, conversation_context, is_active)
            VALUES (:session_token, :created_at, :last_activity, :context, :is_active)
            RETURNING conversation_id
        """
        
        params = {
            "session_token": session_token,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "context": json.dumps({}),
            "is_active": True
        }
        
        result = self.db_handler.execute_query(insert_query, params)
        if result:
            conversation_id = str(result[0][0])
            logging.info(f"ConversationManager: Created new conversation {conversation_id}")
            return conversation_id
        
        raise Exception("Failed to create conversation")
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   sql_query: Optional[str] = None, result_count: Optional[int] = None,
                   intent: Optional[str] = None, entities: Optional[Dict] = None) -> str:
        """
        Add a message to the conversation history.
        
        Args:
            conversation_id: UUID of the conversation
            role: 'user' or 'assistant'  
            content: Message content
            sql_query: Generated SQL query (for assistant messages)
            result_count: Number of results returned (for assistant messages)
            intent: Classified intent ('search', 'refinement', 'clarification')
            entities: Extracted entities from the message
            
        Returns:
            str: message_id (UUID)
        """
        insert_query = """
            INSERT INTO conversation_messages 
            (conversation_id, role, content, sql_query, result_count, intent, entities, created_at)
            VALUES (:conversation_id, :role, :content, :sql_query, :result_count, :intent, :entities, :created_at)
            RETURNING message_id
        """
        
        params = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sql_query": sql_query,
            "result_count": result_count,
            "intent": intent,
            "entities": json.dumps(entities or {}),
            "created_at": datetime.now()
        }
        
        result = self.db_handler.execute_query(insert_query, params)
        if result:
            message_id = str(result[0][0])
            # Update conversation last_activity
            self._update_last_activity(conversation_id)
            return message_id
        
        raise Exception("Failed to add message")
    
    def get_conversation_context(self, conversation_id: str) -> Dict:
        """
        Get the current conversation context.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            Dict: Conversation context data
        """
        query = "SELECT conversation_context FROM conversations WHERE conversation_id = :conversation_id"
        result = self.db_handler.execute_query(query, {"conversation_id": conversation_id})
        
        if result:
            context_json = result[0][0]
            return json.loads(context_json) if context_json else {}
        
        return {}
    
    def update_conversation_context(self, conversation_id: str, context_update: Dict):
        """
        Update conversation context with new information.
        
        Args:
            conversation_id: UUID of the conversation
            context_update: Dictionary with context updates
        """
        current_context = self.get_conversation_context(conversation_id)
        current_context.update(context_update)
        
        update_query = """
            UPDATE conversations 
            SET conversation_context = :context, last_activity = :last_activity
            WHERE conversation_id = :conversation_id
        """
        
        params = {
            "conversation_id": conversation_id,
            "context": json.dumps(current_context),
            "last_activity": datetime.now()
        }
        
        self.db_handler.execute_query(update_query, params)
    
    def get_recent_messages(self, conversation_id: str, limit: int = 5) -> List[Dict]:
        """
        Get recent messages from the conversation.
        
        Args:
            conversation_id: UUID of the conversation
            limit: Maximum number of messages to retrieve
            
        Returns:
            List[Dict]: Recent messages ordered by created_at DESC
        """
        query = """
            SELECT message_id, role, content, sql_query, result_count, intent, entities, created_at
            FROM conversation_messages
            WHERE conversation_id = :conversation_id
            ORDER BY created_at DESC
            LIMIT :limit
        """
        
        result = self.db_handler.execute_query(query, {
            "conversation_id": conversation_id,
            "limit": limit
        })
        
        if not result:
            return []
        
        messages = []
        for row in result:
            message = {
                "message_id": str(row[0]),
                "role": row[1],
                "content": row[2],
                "sql_query": row[3],
                "result_count": row[4],
                "intent": row[5],
                "entities": json.loads(row[6]) if row[6] else {},
                "created_at": row[7]
            }
            messages.append(message)
        
        # Return in chronological order (oldest first)
        return list(reversed(messages))
    
    def classify_intent(self, user_input: str, context: Dict, recent_messages: List[Dict]) -> str:
        """
        Classify user intent based on input and conversation context.
        
        Args:
            user_input: User's message
            context: Current conversation context
            recent_messages: Recent conversation messages
            
        Returns:
            str: Classified intent ('search', 'refinement', 'clarification', 'follow_up')
        """
        user_input_lower = user_input.lower()
        
        # Check for refinement keywords
        refinement_keywords = [
            'more', 'different', 'instead', 'other', 'also', 'too', 'actually',
            'what about', 'how about', 'any', 'show me', 'find', 'not',
            'exclude', 'without', 'except', 'change', 'switch'
        ]
        
        # Check for follow-up keywords
        follow_up_keywords = [
            'when', 'where', 'what time', 'how much', 'cost', 'price',
            'details', 'more info', 'tell me about', 'address', 'location'
        ]
        
        # Check if there are previous assistant messages with results
        has_previous_results = any(
            msg['role'] == 'assistant' and msg.get('result_count', 0) > 0 
            for msg in recent_messages
        )
        
        if has_previous_results:
            if any(keyword in user_input_lower for keyword in refinement_keywords):
                return 'refinement'
            elif any(keyword in user_input_lower for keyword in follow_up_keywords):
                return 'follow_up'
        
        # Default to new search if no previous context or clear new search intent
        return 'search'
    
    def extract_entities(self, user_input: str, context: Dict) -> Dict:
        """
        Extract entities from user input using simple keyword matching.
        
        Args:
            user_input: User's message
            context: Current conversation context
            
        Returns:
            Dict: Extracted entities (dance_style, event_type, time_reference, etc.)
        """
        entities = {}
        user_input_lower = user_input.lower()
        
        # Dance styles (from sql_prompt.txt)
        dance_styles = [
            '2-step', 'argentine tango', 'bachata', 'balboa', 'cha cha', 'cha cha cha',
            'country waltz', 'double shuffle', 'douceur', 'east coast swing', 'foxtrot',
            'kizomba', 'lindy', 'lindy hop', 'line dance', 'merengue', 'milonga',
            'night club', 'nite club', 'nite club 2', 'nite club two', 'quickstep',
            'rhumba', 'rumba', 'salsa', 'samba', 'semba', 'swing', 'tango',
            'tarraxa', 'tarraxinha', 'tarraxo', 'two step', 'urban kiz', 'waltz',
            'wcs', 'west coast swing', 'zouk'
        ]
        
        found_styles = [style for style in dance_styles if style in user_input_lower]
        if found_styles:
            entities['dance_style'] = found_styles
        
        # Event types
        event_types = ['social dance', 'class', 'workshop', 'live music', 'lesson']
        found_types = [event_type for event_type in event_types if event_type in user_input_lower]
        if found_types:
            entities['event_type'] = found_types
        elif 'learn' in user_input_lower:
            entities['event_type'] = ['class', 'workshop']
        
        # Time references
        time_keywords = {
            'today': 'today',
            'tonight': 'today', 
            'tomorrow': 'tomorrow',
            'this week': 'this_week',
            'next week': 'next_week',
            'this weekend': 'weekend',
            'weekend': 'weekend'
        }
        
        for keyword, time_ref in time_keywords.items():
            if keyword in user_input_lower:
                entities['time_reference'] = time_ref
                break
        
        return entities
    
    def _update_last_activity(self, conversation_id: str):
        """Update last_activity timestamp for a conversation."""
        query = "UPDATE conversations SET last_activity = :last_activity WHERE conversation_id = :conversation_id"
        self.db_handler.execute_query(query, {
            "conversation_id": conversation_id,
            "last_activity": datetime.now()
        })
    
    def cleanup_inactive_conversations(self, days_old: int = 7) -> int:
        """
        Clean up conversations older than specified days.
        
        Args:
            days_old: Number of days after which conversations are considered inactive
            
        Returns:
            int: Number of conversations cleaned up
        """
        cleanup_query = """
            DELETE FROM conversations 
            WHERE last_activity < :cutoff_date
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        result = self.db_handler.execute_query(cleanup_query, {"cutoff_date": cutoff_date})
        
        # The execute_query method returns rowcount for non-SELECT queries
        cleaned_count = result if isinstance(result, int) else 0
        logging.info(f"ConversationManager: Cleaned up {cleaned_count} inactive conversations")
        return cleaned_count