import unittest
from unittest.mock import Mock, patch, MagicMock

from mcpx_eval import Judge, Model, Score, Results, Database
from mcpx_eval.models import ScoreModel
from mcpx_eval.judge import ToolAnalysis


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.judge = Judge(
            models=["test-model"],
            judge_model="test-judge",
            ignore_tools=["ignored-tool"],
            client=MagicMock(),
        )

    def test_add_model(self):
        """Test adding models to the judge"""
        judge = Judge(
            client=MagicMock(),
        )

        # Test adding string model
        judge.add_model("gpt-4")
        self.assertEqual(len(judge.models), 1)
        self.assertEqual(judge.models[0].name, "gpt-4")

        # Test adding Model instance
        model = Model(name="anthropic:claude-3")
        judge.add_model(model)
        self.assertEqual(len(judge.models), 2)
        self.assertEqual(judge.models[1].name, "claude-3")
        self.assertEqual(judge.models[1].provider, "anthropic")

        # Test adding model with profile
        judge.add_model("mistral", profile="custom")
        self.assertEqual(len(judge.models), 3)
        self.assertEqual(judge.models[2].name, "mistral")
        self.assertEqual(judge.models[2].profile, "custom")


class TestToolAnalysis(unittest.TestCase):
    def test_analyze_message_unique_tools(self):
        """Test analyzing unique tool calls"""
        from mcpx_eval.judge import ToolAnalysis

        tool_analysis = ToolAnalysis()

        # Test first unique tool call
        msg1 = {"tool": {"name": "test_tool", "input": {"param": "value1"}}}
        tool_analysis.analyze_message(msg1, 0)

        self.assertEqual(tool_analysis.total_tool_calls, 1)
        self.assertEqual(tool_analysis.redundant_tool_calls, 0)
        self.assertEqual(tool_analysis.tool_analysis["tool_0"]["redundancy"], "unique")

        # Test second unique tool call
        msg2 = {"tool": {"name": "test_tool", "input": {"param": "value2"}}}
        tool_analysis.analyze_message(msg2, 1)

        self.assertEqual(tool_analysis.total_tool_calls, 2)
        self.assertEqual(tool_analysis.redundant_tool_calls, 0)

    def test_analyze_message_redundant_tools(self):
        """Test analyzing redundant tool calls"""
        from mcpx_eval.judge import ToolAnalysis

        tool_analysis = ToolAnalysis()

        # Add first tool call
        msg1 = {"tool": {"name": "test_tool", "input": {"param": "value1"}}}
        tool_analysis.analyze_message(msg1, 0)

        # Add redundant tool call
        msg2 = {"tool": {"name": "test_tool", "input": {"param": "value1"}}}
        tool_analysis.analyze_message(msg2, 1)

        self.assertEqual(tool_analysis.total_tool_calls, 2)
        self.assertEqual(tool_analysis.redundant_tool_calls, 1)
        self.assertEqual(
            tool_analysis.tool_analysis["tool_1"]["redundancy"], "redundant"
        )


class TestModelApiConfig(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "OPENAI_HOST": "https://custom-openai.com",
            "GPT-4_HOST": "https://custom-gpt4.com",
        },
    )
    def test_get_host_url(self):
        """Test getting host URLs for different providers"""
        from mcpx_eval.judge import ModelApiConfig

        # Test OpenAI default
        url = ModelApiConfig.get_host_url("gpt-3.5-turbo", "openai")
        self.assertEqual(url, "https://custom-openai.com/v1")

        # Test model-specific override
        url = ModelApiConfig.get_host_url("gpt-4", "openai")
        self.assertEqual(url, "https://custom-gpt4.com/v1")

        # Test Ollama default
        url = ModelApiConfig.get_host_url("llama2", "ollama")
        self.assertEqual(url, "http://127.0.0.1:11434/v1")


class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items
        self.index = 0

    async def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = self.items[self.index]
        except IndexError:
            raise StopAsyncIteration
        self.index += 1
        return item


class MockPart:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MockResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestJudgeEvaluation(unittest.IsolatedAsyncioTestCase):
    @patch("mcpx_eval.judge.Chat")
    @patch("mcpx_eval.judge.mcp_run")
    async def test_evaluate_model_success(self, mock_mcp_run, mock_chat):
        """Test successful model evaluation"""
        # Setup mock mcp_run.Client with proper tools attribute
        mock_tools = MagicMock()
        mock_tools.keys.return_value = ["test_tool"]
        mock_client = MagicMock()
        mock_client.tools = mock_tools
        mock_mcp_run.Client = Mock(return_value=mock_client)
        mock_mcp_run.ClientConfig = Mock()

        # Setup mock chat instance
        mock_chat_instance = MagicMock()
        mock_chat_instance.client = mock_client

        # Setup response parts
        model_response_parts = [
            MockPart(part_kind="text", content="Test response"),
            MockPart(
                part_kind="tool-call",
                tool_name="test_tool",
                tool_call_id="123",
                args={"param": "value"},
                args_as_dict=lambda: {"param": "value"},
            ),
        ]
        request_parts = [
            MockPart(
                part_kind="tool-return",
                tool_name="test_tool",
                tool_call_id="123",
                content="Tool result",
            )
        ]

        async def mock_iter(prompt):
            yield MockResponse(model_response=MockResponse(parts=model_response_parts))
            yield MockResponse(request=MockResponse(parts=request_parts))
            yield MockResponse(data=MockPart(data="Final result"))

        mock_chat_instance.iter = mock_iter
        mock_chat.return_value = mock_chat_instance

        judge = Judge(client=MagicMock())
        model = Model(name="test-model")
        tool_analysis = ToolAnalysis()

        result = await judge.evaluate_model(model, "Test prompt", tool_analysis)

        self.assertIsNotNone(result)
        self.assertEqual(
            len(result["messages"]), 4
        )  # text, tool-call, tool-return, final_result
        self.assertEqual(result["messages"][0]["kind"], "text")
        self.assertEqual(result["messages"][1]["kind"], "tool-call")
        self.assertEqual(result["messages"][2]["kind"], "tool-return")
        self.assertEqual(result["messages"][3]["kind"], "final_result")

    @patch("mcpx_eval.judge.Chat")
    @patch("mcpx_eval.judge.mcp_run")
    async def test_evaluate_model_failure(self, mock_mcp_run, mock_chat):
        """Test model evaluation with error"""
        # Setup mock mcp_run.Client
        mock_client = Mock()
        mock_mcp_run.Client = Mock(return_value=mock_client)
        mock_mcp_run.ClientConfig = Mock()

        mock_chat_instance = Mock()

        async def mock_iter(prompt):
            raise Exception("Test error")
            yield  # Needed to make it a generator

        mock_chat_instance.iter = mock_iter
        mock_chat.return_value = mock_chat_instance

        judge = Judge(client=MagicMock())
        model = Model(name="test-model")
        tool_analysis = ToolAnalysis()

        result = await judge.evaluate_model(model, "Test prompt", tool_analysis)

        self.assertIsNone(result)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")  # Use in-memory SQLite for testing

    def test_save_and_retrieve_results(self):
        """Test saving and retrieving test results"""
        # Create test data
        test_name = "test1"
        score_data = ScoreModel(
            tool_use=80,
            accuracy=90,
            completeness=85,
            quality=88,
            hallucination_score=5,
            false_claims=["claim1"],
            llm_output="test output",
            description="test description",
        )

        score = Score(
            score=score_data,
            model="test-model",
            duration=1.5,
            tool_analysis={"tool_1": {"name": "test_tool", "redundancy": "unique"}},
            redundant_tool_calls=0,
            tool_calls=1,
        )

        results = Results(scores=[score], duration=1.5)

        # Save results
        self.db.save_results(test_name, results)

        # Retrieve and verify results
        retrieved = self.db.average_results(test_name)

        self.assertEqual(len(retrieved.scores), 1)
        self.assertEqual(retrieved.scores[0].model, "test-model")
        self.assertEqual(retrieved.scores[0].duration, 1.5)
        self.assertEqual(retrieved.scores[0].tool_calls, 1)
        self.assertEqual(retrieved.scores[0].redundant_tool_calls, 0)
        self.assertEqual(retrieved.scores[0].accuracy, 90)


if __name__ == "__main__":
    unittest.main()
