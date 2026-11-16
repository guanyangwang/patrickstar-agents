import json
import tempfile
import unittest
from pathlib import Path

import httpx

from patrickstar_agents.config import ConfigError
from patrickstar_agents.mathos_api import BASE_URL, MathOSAPIProvider
from patrickstar_agents.providers import ProviderError


class MathOSAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_request_model_selection_and_usage(self):
        requests = []

        def handler(request):
            requests.append(request)
            self.assertEqual(str(request.url), BASE_URL + "/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer nuc_ak_test")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "model/b")
            self.assertFalse(body["stream"])
            self.assertNotIn("previous_response_id", body)
            return httpx.Response(
                200,
                json={
                    "model": "actual-b",
                    "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            base = MathOSAPIProvider(
                "nuc_ak_test",
                [{"id": "model/a"}, {"id": "model/b", "name": "B"}],
                "model/a",
                client=client,
            )
            base.model = "model/b"
            result = await base.complete("2+2?", system="Be concise.")
            self.assertEqual(result.model, "actual-b")
            self.assertEqual(result.usage["total_tokens"], 7)
            self.assertEqual(result.usage["route"], "mathos-model-api")
            self.assertEqual(len(requests), 1)

    async def test_failures_are_sanitized_and_not_replayed(self):
        for status, body in [
            (401, {"error": "secret"}),
            (429, {"error": "secret"}),
            (500, {"error": "secret"}),
            (
                200,
                {
                    "choices": [
                        {"finish_reason": "length", "message": {"content": "partial"}}
                    ]
                },
            ),
        ]:
            count = 0

            def handler(request, status=status, body=body):
                nonlocal count
                count += 1
                return httpx.Response(status, json=body)

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                p = MathOSAPIProvider(
                    "nuc_ak_test", [{"id": "model/a"}], "model/a", client=client
                )
                with self.assertRaises(ProviderError) as caught:
                    await p.complete("x", system="y")
                self.assertNotIn("secret", str(caught.exception))
                self.assertNotIn("partial", str(caught.exception))
                self.assertEqual(count, 1)

    def test_private_configuration_and_no_invalid_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mathos-api.json"
            data = {
                "enabled": True,
                "api_key": "nuc_ak_test",
                "models": [{"id": "a"}],
                "model": "a",
            }
            path.write_text(json.dumps(data))
            path.chmod(0o600)
            self.assertIsInstance(MathOSAPIProvider.from_file(path), MathOSAPIProvider)
            data["model"] = "unverified"
            path.write_text(json.dumps(data))
            with self.assertRaises(ConfigError):
                MathOSAPIProvider.from_file(path)
