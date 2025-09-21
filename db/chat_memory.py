"""
Chat Memory Store for maintaining conversation context in RAG system.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class ChatMemoryStore:
    """Simple in-memory chat store with persistent backup."""
    
    def __init__(self, memory_directory: str = "./chat_memory", max_messages: int = 20):
        self.memory_directory = Path(memory_directory)
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        
        # Create memory directory if it doesn't exist
        self.memory_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory if available
        self._load_memory()
    
    def add_user_message(self, message: str):
        """Add a user message to memory."""
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_memory()
        self._save_memory()
    
    def add_assistant_message(self, message: str):
        """Add an assistant response to memory."""
        self.messages.append({
            "role": "assistant", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_memory()
        self._save_memory()
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation context for the prompt."""
        if not self.messages:
            return "No previous conversation context."
        
        # Get last num_turns * 2 messages (user + assistant pairs)
        recent_messages = self.messages[-(num_turns * 2):]
        
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict]:
        """Get conversation history with limit."""
        return self.messages[-limit:] if limit > 0 else self.messages
    
    def get_session_stats(self) -> Dict:
        """Get session statistics."""
        return {
            "total_messages": len(self.messages),
            "user_messages": len([msg for msg in self.messages if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in self.messages if msg["role"] == "assistant"]),
            "session_start": self.messages[0]["timestamp"] if self.messages else None
        }
    
    def clear_session(self):
        """Clear current session."""
        self.clear_memory()
    
    def list_sessions(self) -> List[str]:
        """List available sessions."""
        # For now, just return the current session
        return ["current"]
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None
    
    def clear_memory(self):
        """Clear all chat memory."""
        self.messages = []
        self._save_memory()
    
    def _trim_memory(self):
        """Keep only the most recent messages."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def _save_memory(self):
        """Save memory to persistent storage."""
        try:
            memory_file = self.memory_directory / "chat_history.json"
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save chat memory: {e}")
    
    def _load_memory(self):
        """Load memory from persistent storage."""
        try:
            memory_file = self.memory_directory / "chat_history.json"
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                self._trim_memory()  # Ensure we don't exceed max_messages
        except Exception as e:
            print(f"Warning: Could not load chat memory: {e}")
            self.messages = []