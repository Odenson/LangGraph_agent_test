import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the 'scr' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scr')))

import main

class TestMain(unittest.TestCase):
    def test_state_class(self):
        # Test that State is a TypedDict with 'messages'
        self.assertIn('messages', main.State.__annotations__)

    @patch('main.llm_with_tools')
    def test_chatbot_returns_message(self, mock_llm_with_tools):
        # Mock the llm_with_tools.invoke to return a dummy message
        dummy_message = MagicMock()
        dummy_message.invoke.return_value = "Test response"
        mock_llm_with_tools.invoke.return_value = "Test response"
        state = {"messages": ["Hello"]}
        result = main.chatbot(state)
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], "Test response")

    def test_graph_builder_nodes(self):
        # Test that nodes are added to the graph_builder
        nodes = main.graph_builder.nodes
        self.assertIn("chatbot", nodes)
        self.assertIn("tools", nodes)

if __name__ == "__main__":
    unittest.main()