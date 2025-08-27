import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChatMemoryStore:
    def __init__(self, memory_directory: str = "./chat_memory"):
        """Initialize chat memory store with persistent storage"""
        self.memory_directory = Path(memory_directory)
        self.memory_directory.mkdir(exist_ok=True)
        
        # Create session file path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_file = self.memory_directory / f"chat_session_{self.session_id}.json"
        
        # Initialize session
        self.conversation_history = []
        self._initialize_session()
        
        logger.info(f"Chat memory initialized for session: {self.session_id}")
    
    def _initialize_session(self):
        """Initialize new chat session"""
        session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "conversation": []
        }
        
        # Save initial session file
        with open(self.current_session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    def add_message(self, message: str, role: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history"""
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # 'user' or 'assistant'
            "message": message,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message_entry)
        self._save_session()
        
        logger.debug(f"Added {role} message to chat memory")
    
    def add_user_message(self, message: str, metadata: Optional[Dict] = None):
        """Add a user message to memory"""
        self.add_message(message, "user", metadata)
    
    def add_assistant_message(self, message: str, metadata: Optional[Dict] = None):
        """Add an assistant message to memory"""
        self.add_message(message, "assistant", metadata)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history with optional limit"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def get_context_for_llm(self, context_window: int = 10) -> str:
        """Get formatted context for LLM with recent conversation"""
        recent_messages = self.get_conversation_history(limit=context_window)
        
        if not recent_messages:
            return ""
        
        context_lines = ["Previous conversation context:"]
        for msg in recent_messages:
            role = msg['role'].capitalize()
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M")
            context_lines.append(f"[{timestamp}] {role}: {msg['message']}")
        
        return "\n".join(context_lines)
    
    def _save_session(self):
        """Save current session to file"""
        session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": len(self.conversation_history),
            "conversation": self.conversation_history
        }
        
        with open(self.current_session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    def clear_session(self):
        """Clear current session and start fresh"""
        self.conversation_history = []
        self._initialize_session()
        logger.info(f"Chat session cleared: {self.session_id}")
    
    def load_previous_session(self, session_id: str) -> bool:
        """Load a previous chat session"""
        session_file = self.memory_directory / f"chat_session_{session_id}.json"
        
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                self.session_id = session_data['session_id']
                self.conversation_history = session_data.get('conversation', [])
                self.current_session_file = session_file
                
                logger.info(f"Loaded previous session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return False
        
        logger.warning(f"Session file not found: {session_id}")
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available chat sessions"""
        sessions = []
        
        for session_file in self.memory_directory.glob("chat_session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    "session_id": session_data['session_id'],
                    "created_at": session_data['created_at'],
                    "message_count": session_data.get('message_count', 0),
                    "file_path": str(session_file)
                })
            except Exception as e:
                logger.error(f"Failed to read session file {session_file}: {e}")
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        return sessions
    
    def cleanup_old_sessions(self, keep_last_n: int = 10):
        """Keep only the N most recent sessions"""
        sessions = self.list_sessions()
        
        if len(sessions) > keep_last_n:
            sessions_to_delete = sessions[keep_last_n:]
            
            for session in sessions_to_delete:
                try:
                    os.remove(session['file_path'])
                    logger.info(f"Deleted old session: {session['session_id']}")
                except Exception as e:
                    logger.error(f"Failed to delete session {session['session_id']}: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session"""
        user_messages = sum(1 for msg in self.conversation_history if msg['role'] == 'user')
        assistant_messages = sum(1 for msg in self.conversation_history if msg['role'] == 'assistant')
        
        return {
            "session_id": self.session_id,
            "total_messages": len(self.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "session_duration": self._get_session_duration(),
            "file_path": str(self.current_session_file)
        }
    
    def _get_session_duration(self) -> str:
        """Calculate session duration"""
        if not self.conversation_history:
            return "0 minutes"
        
        start_time = datetime.fromisoformat(self.conversation_history[0]['timestamp'])
        end_time = datetime.fromisoformat(self.conversation_history[-1]['timestamp'])
        duration = end_time - start_time
        
        minutes = duration.total_seconds() / 60
        return f"{int(minutes)} minutes"
