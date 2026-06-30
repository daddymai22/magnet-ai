import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import torch
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    """Single message in conversation"""
    role: MessageRole
    content: str
    timestamp: datetime
    feedback: Optional[str] = None  # "like" or "dislike"
    
    def to_dict(self) -> dict:
        return {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'feedback': self.feedback
        }

@dataclass
class Conversation:
    """Represents a single conversation session"""
    conversation_id: str
    agent_id: int
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    title: str = "New Conversation"
    
    def add_message(self, message: Message) -> None:
        """Add message to conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def add_feedback(self, message_index: int, feedback_type: str) -> bool:
        """Add feedback (like/dislike) to a message"""
        if message_index < 0 or message_index >= len(self.messages):
            return False
        self.messages[message_index].feedback = feedback_type
        return True
    
    def get_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context for model input"""
        recent = self.messages[-max_messages:]
        context = ""
        for msg in recent:
            context += f"{msg.role.value.upper()}: {msg.content}\n\n"
        return context
    
    def to_dict(self) -> dict:
        return {
            'conversation_id': self.conversation_id,
            'agent_id': self.agent_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self.messages),
            'messages': [msg.to_dict() for msg in self.messages]
        }

class ConversationManager:
    """Manages conversation persistence and retrieval"""
    
    def __init__(self, conversation_dir: str = "./conversations"):
        self.conversation_dir = Path(conversation_dir)
        self.conversation_dir.mkdir(exist_ok=True)
        self.conversations: Dict[str, Conversation] = {}
        self._load_conversations()
    
    def _load_conversations(self) -> None:
        """Load all saved conversations from disk"""
        for conv_file in self.conversation_dir.glob("*.json"):
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                    
                messages = [
                    Message(
                        role=MessageRole(msg['role']),
                        content=msg['content'],
                        timestamp=datetime.fromisoformat(msg['timestamp']),
                        feedback=msg.get('feedback')
                    )
                    for msg in data['messages']
                ]
                
                conversation = Conversation(
                    conversation_id=data['conversation_id'],
                    agent_id=data['agent_id'],
                    messages=messages,
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at']),
                    title=data.get('title', 'Loaded Conversation')
                )
                
                self.conversations[conversation.conversation_id] = conversation
                logger.info(f"Loaded conversation: {conversation.conversation_id}")
            
            except Exception as e:
                logger.error(f"Error loading {conv_file}: {e}")
    
    def create_conversation(self, conversation_id: str, agent_id: int, title: str = "") -> Conversation:
        """Create new conversation"""
        conversation = Conversation(
            conversation_id=conversation_id,
            agent_id=agent_id,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            title=title or f"Chat with Agent {agent_id}"
        )
        self.conversations[conversation_id] = conversation
        self._save_conversation(conversation)
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation"""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self, agent_id: Optional[int] = None) -> List[Dict]:
        """List all conversations, optionally filtered by agent"""
        convs = self.conversations.values()
        if agent_id is not None:
            convs = [c for c in convs if c.agent_id == agent_id]
        
        return [
            {
                'conversation_id': c.conversation_id,
                'agent_id': c.agent_id,
                'title': c.title,
                'message_count': len(c.messages),
                'updated_at': c.updated_at.isoformat()
            }
            for c in sorted(convs, key=lambda x: x.updated_at, reverse=True)
        ]
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to disk (live saving)"""
        file_path = self.conversation_dir / f"{conversation.conversation_id}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def add_message_to_conversation(self, conversation_id: str, message: Message) -> bool:
        """Add message and save immediately (live saving)"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation.add_message(message)
        self._save_conversation(conversation)  # Live save
        return True
    
    def add_feedback(self, conversation_id: str, message_index: int, feedback_type: str) -> bool:
        """Add feedback to message and save"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        if conversation.add_feedback(message_index, feedback_type):
            self._save_conversation(conversation)  # Live save
            return True
        return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id not in self.conversations:
            return False
        
        del self.conversations[conversation_id]
        file_path = self.conversation_dir / f"{conversation_id}.json"
        if file_path.exists():
            file_path.unlink()
        return True

class SimpleLLMChat:
    """Simple LLM chat interface (placeholder for actual model inference)"""
    
    def __init__(self, model_name: str = "magnet-base-v1"):
        self.model_name = model_name
        self.context_window = 2048
        self.temperature = 0.7
        logger.info(f"Initialized chat with model: {model_name}")
    
    async def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response (placeholder - would load actual model)"""
        # This is a placeholder. In production, this would:
        # 1. Load the trained model
        # 2. Tokenize the input
        # 3. Generate tokens with temperature sampling
        # 4. Decode back to text
        
        await asyncio.sleep(0.1)  # Simulate generation time
        
        responses = [
            "I've analyzed your request and here's what I found...",
            "That's an interesting question! Let me think about this...",
            "Based on the context, I would suggest...",
            "I understand. Here's my approach to solving this:"
        ]
        
        import random
        return random.choice(responses) + " [Placeholder response]"

class ChatUI:
    """Command-line chat interface (can be extended to web UI)"""
    
    def __init__(self, agent_id: int = 0):
        self.agent_id = agent_id
        self.conversation_manager = ConversationManager()
        self.llm_chat = SimpleLLMChat()
        self.current_conversation: Optional[Conversation] = None
    
    async def start_new_conversation(self, title: str = "") -> Conversation:
        """Start a new conversation"""
        conv_id = f"conv_{self.agent_id}_{datetime.now().timestamp()}"
        self.current_conversation = self.conversation_manager.create_conversation(
            conv_id, self.agent_id, title
        )
        print(f"\n✨ Started new conversation: {self.current_conversation.title}")
        return self.current_conversation
    
    async def send_message(self, user_input: str) -> None:
        """Send message and get response"""
        if not self.current_conversation:
            print("❌ No active conversation. Start one with 'new'.")
            return
        
        # Add user message
        user_msg = Message(
            role=MessageRole.USER,
            content=user_input,
            timestamp=datetime.now()
        )
        self.conversation_manager.add_message_to_conversation(
            self.current_conversation.conversation_id,
            user_msg
        )
        print(f"\n👤 You: {user_input}")
        
        # Get context
        context = self.current_conversation.get_context()
        
        # Generate response
        print("\n🤖 Agent is thinking...", end="", flush=True)
        response = await self.llm_chat.generate_response(user_input, context)
        print("\r" + " " * 30 + "\r", end="")  # Clear thinking message
        
        # Add assistant message
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response,
            timestamp=datetime.now()
        )
        self.conversation_manager.add_message_to_conversation(
            self.current_conversation.conversation_id,
            assistant_msg
        )
        print(f"🧲 Agent: {response}")
    
    async def add_feedback(self, feedback_type: str) -> None:
        """Add like/dislike to last assistant message"""
        if not self.current_conversation or len(self.current_conversation.messages) < 2:
            print("❌ No message to provide feedback on.")
            return
        
        # Find last assistant message
        for i in range(len(self.current_conversation.messages) - 1, -1, -1):
            if self.current_conversation.messages[i].role == MessageRole.ASSISTANT:
                self.conversation_manager.add_feedback(
                    self.current_conversation.conversation_id,
                    i,
                    feedback_type
                )
                emoji = "👍" if feedback_type == "like" else "👎"
                print(f"{emoji} Feedback recorded!")
                return
    
    def list_conversations(self) -> None:
        """List all conversations for this agent"""
        convs = self.conversation_manager.list_conversations(self.agent_id)
        if not convs:
            print("No conversations yet.")
            return
        
        print("\n📚 Your Conversations:")
        for i, conv in enumerate(convs, 1):
            print(f"{i}. {conv['title']} ({conv['message_count']} messages)")
    
    async def load_conversation(self, conversation_id: str) -> bool:
        """Load a saved conversation"""
        conv = self.conversation_manager.get_conversation(conversation_id)
        if not conv:
            print("❌ Conversation not found.")
            return False
        
        self.current_conversation = conv
        print(f"\n📖 Loaded conversation: {conv.title}")
        return True

async def main():
    """CLI chat interface demo"""
    print("="*50)
    print("🧲 MAGNET AI - Chat Interface")
    print("="*50)
    print("Commands: 'new', 'list', 'load <id>', 'like', 'dislike', 'exit'\n")
    
    chat_ui = ChatUI(agent_id=0)
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "new":
                title = input("Conversation title (or press Enter): ").strip()
                await chat_ui.start_new_conversation(title)
            
            elif command == "list":
                chat_ui.list_conversations()
            
            elif command.startswith("load "):
                conv_id = command[5:].strip()
                await chat_ui.load_conversation(conv_id)
            
            elif command == "like":
                await chat_ui.add_feedback("like")
            
            elif command == "dislike":
                await chat_ui.add_feedback("dislike")
            
            elif command == "exit":
                print("\n👋 Goodbye!")
                break
            
            elif command.strip():
                await chat_ui.send_message(command)
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
