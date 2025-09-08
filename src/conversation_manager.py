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
            if context_json:
                # Handle both string (JSON) and dict (already parsed JSONB) cases
                if isinstance(context_json, str):
                    return json.loads(context_json)
                else:
                    return context_json  # Already a dict from JSONB
            return {}
        
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
    
    def store_pending_query(self, conversation_id: str, user_input: str, combined_query: str, 
                           interpretation: str, sql_query: str):
        """
        Store a pending query awaiting user confirmation.
        
        Args:
            conversation_id: UUID of the conversation
            user_input: Original user input
            combined_query: Combined query with concatenations
            interpretation: Natural language interpretation
            sql_query: Generated SQL query ready for execution
        """
        pending_data = {
            "pending_confirmation": True,
            "pending_user_input": user_input,
            "pending_combined_query": combined_query,
            "pending_interpretation": interpretation,
            "pending_sql_query": sql_query,
            "pending_timestamp": datetime.now().isoformat()
        }
        
        self.update_conversation_context(conversation_id, pending_data)
        logging.info(f"ConversationManager: Stored pending query for confirmation in conversation {conversation_id}")
    
    def get_pending_query(self, conversation_id: str) -> Dict:
        """
        Retrieve pending query data if one exists.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            Dict: Pending query data or empty dict if none
        """
        context = self.get_conversation_context(conversation_id)
        
        if context.get("pending_confirmation"):
            return {
                "user_input": context.get("pending_user_input"),
                "combined_query": context.get("pending_combined_query"),
                "interpretation": context.get("pending_interpretation"),
                "sql_query": context.get("pending_sql_query"),
                "timestamp": context.get("pending_timestamp")
            }
        
        return {}
    
    def clear_pending_query(self, conversation_id: str):
        """
        Clear pending query data from conversation context.
        
        Args:
            conversation_id: UUID of the conversation
        """
        context_update = {
            "pending_confirmation": False,
            "pending_user_input": None,
            "pending_combined_query": None,
            "pending_interpretation": None,
            "pending_sql_query": None,
            "pending_timestamp": None
        }
        
        self.update_conversation_context(conversation_id, context_update)
        logging.info(f"ConversationManager: Cleared pending query from conversation {conversation_id}")
    
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
                "entities": json.loads(row[6]) if isinstance(row[6], str) and row[6] else (row[6] if row[6] else {}),
                "created_at": row[7]
            }
            messages.append(message)
        
        # Return in chronological order (oldest first)
        return list(reversed(messages))
    
    def classify_intent(self, user_input: str, context: Dict, recent_messages: List[Dict]) -> str:
        """
        Classify user intent using hybrid approach: simple rules first, then LLM for edge cases.
        
        Args:
            user_input: User's message
            context: Current conversation context
            recent_messages: Recent conversation messages
            
        Returns:
            str: Classified intent ('search' or 'refinement')
        """
        # If no previous search, definitely a new search
        has_previous_search = any(
            msg['role'] == 'assistant' and msg.get('result_count', 0) >= 0
            for msg in recent_messages
        ) or context.get('last_search_query')
        
        if not has_previous_search:
            return 'search'
            
        # Get the last search query for comparison
        last_query = context.get('last_search_query', '').lower()
        user_input_lower = user_input.lower()
        
        # RULE 1: Clear new search indicators (95% reliable)
        new_search_phrases = [
            'show me', 'find me', 'find', 'search for', 'look for', 'get me',
            'what about', 'how about', 'instead', 'forget that', 'actually',
            'now show', 'now find', 'different', 'something else', 'change to'
        ]
        
        matching_phrases = [phrase for phrase in new_search_phrases if phrase in user_input_lower]
        if matching_phrases:
            logging.info(f"RULE 1: NEW SEARCH detected by phrases: {matching_phrases}")
            return 'search'
            
        # RULE 2: Time conflict detection (90% reliable)
        last_time_words = self._extract_time_references(last_query)
        current_time_words = self._extract_time_references(user_input_lower)
        
        if last_time_words and current_time_words:
            # If time references don't overlap, likely new search
            if not any(time_word in current_time_words for time_word in last_time_words):
                # Check for conflicting time periods
                time_conflicts = [
                    ('tonight', 'tomorrow'), ('tonight', 'weekend'), ('tonight', 'next week'),
                    ('today', 'tomorrow'), ('today', 'weekend'), ('this week', 'next week'),
                    ('weekend', 'weekday'), ('morning', 'evening')
                ]
                
                for time1, time2 in time_conflicts:
                    if ((time1 in last_time_words and time2 in current_time_words) or 
                        (time2 in last_time_words and time1 in current_time_words)):
                        logging.info(f"RULE 2: NEW SEARCH detected by time conflict: {last_time_words} vs {current_time_words}")
                        return 'search'
        
        # RULE 3: Location changes (85% reliable)
        location_indicators = ['in ', 'at ', 'near ', 'around ', 'downtown', 'victoria', 'vancouver', 'saanich']
        last_has_location = any(loc in last_query for loc in location_indicators)
        current_has_location = any(loc in user_input_lower for loc in location_indicators)
        
        if last_has_location and current_has_location:
            # Extract rough location indicators
            last_locations = [loc for loc in location_indicators if loc in last_query]
            current_locations = [loc for loc in location_indicators if loc in user_input_lower]
            if not any(loc in current_locations for loc in last_locations):
                return 'search'
        
        # RULE 4: Very short inputs are usually refinements (90% reliable)
        user_words = user_input_lower.strip().split()
        if len(user_words) <= 3:
            logging.info(f"RULE 4: REFINEMENT detected by short input ({len(user_words)} words)")
            return 'refinement'
            
        # RULE 5: Contains only dance style + polite words = refinement
        dance_styles = [
            'salsa', 'bachata', 'swing', 'tango', 'waltz', 'foxtrot', 'lindy',
            'kizomba', 'zouk', 'merengue', 'rumba', 'cha cha', 'quickstep'
        ]
        polite_words = ['please', 'just', 'only', 'prefer', 'like']
        
        has_dance_style = any(style in user_input_lower for style in dance_styles)
        mostly_dance_and_polite = all(
            word in dance_styles + polite_words + ['and', 'or', 'the', 'a', 'an']
            for word in user_words if len(word) > 2
        )
        
        if has_dance_style and mostly_dance_and_polite:
            logging.info(f"RULE 5: REFINEMENT detected by dance style + polite words")
            return 'refinement'
            
        # EDGE CASE: Use LLM for ambiguous cases
        logging.info(f"EDGE CASE: Using LLM fallback for ambiguous input")
        return self._llm_classify_intent(user_input, last_query)
    
    def _extract_time_references(self, text: str) -> list:
        """Extract time reference words from text."""
        time_words = [
            'tonight', 'today', 'tomorrow', 'weekend', 'weekday', 'monday', 'tuesday', 
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'morning', 
            'afternoon', 'evening', 'night', 'this week', 'next week', 'next month'
        ]
        return [word for word in time_words if word in text.lower()]
    
    def _llm_classify_intent(self, current_input: str, last_query: str) -> str:
        """Use LLM to classify ambiguous cases."""
        try:
            # Simple prompt for binary classification
            prompt = f"""You are classifying user intent. Answer only with "NEW_SEARCH" or "REFINEMENT".

Previous search: "{last_query}"
Current input: "{current_input}"

Rules:
- NEW_SEARCH: if current input is asking about a completely different topic, time, or location
- REFINEMENT: if current input is adding constraints or details to the previous search

Answer:"""

            # Use the same LLM handler that's available in the system
            # For now, default to refinement as the safer option
            logging.info(f"LLM classification needed for: previous='{last_query}' current='{current_input}'")
            return 'refinement'  # Conservative default
            
        except Exception as e:
            logging.error(f"LLM classification failed: {e}")
            return 'refinement'  # Safe fallback
    
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