"""
Tests for app.schemas — Pydantic request/response models.

Run: .venv/bin/python -m tests.test_schemas
"""

from app.schemas import Message, ChatRequest, ChatResponse, Recommendation


def test_valid_chat_request():
    req = ChatRequest(messages=[
        Message(role="user", content="I need a Java assessment"),
        Message(role="assistant", content="What seniority level?"),
        Message(role="user", content="Mid-level"),
    ])
    print(f"[PASS] ChatRequest: {len(req.messages)} messages")


def test_valid_chat_response_with_recommendations():
    resp = ChatResponse(
        reply="Here are assessments for a mid-level Java developer.",
        recommendations=[
            Recommendation(
                name="Core Java (Advanced Level) (New)",
                url="https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                test_type="K"
            ),
        ],
        end_of_conversation=False,
    )
    print(f"[PASS] ChatResponse: {len(resp.recommendations)} recommendations")
    print(resp.model_dump_json(indent=2))


def test_empty_recommendations_for_clarification():
    resp = ChatResponse(
        reply="What seniority level is this role?",
        recommendations=[],
        end_of_conversation=False,
    )
    print(f"[PASS] Empty recommendations: {resp.recommendations}")


def test_invalid_test_type_raises_error():
    try:
        Recommendation(name="Test", url="https://www.shl.com/test/", test_type="X")
        print("[FAIL] Should have raised ValueError")
    except Exception as e:
        print(f"[PASS] Caught invalid test_type: {e}")


def test_multi_code_test_type():
    multi = Recommendation(
        name="Excel 365",
        url="https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
        test_type="K,S"
    )
    print(f"[PASS] Multi test_type: {multi.test_type}")


def test_too_many_recommendations_raises_error():
    try:
        recs = [
            Recommendation(
                name=f"Test {i}",
                url=f"https://www.shl.com/t/{i}/",
                test_type="K"
            )
            for i in range(11)
        ]
        ChatResponse(reply="Too many", recommendations=recs, end_of_conversation=False)
        print("[FAIL] Should have raised ValueError")
    except Exception as e:
        print(f"[PASS] Caught too many recommendations: {e}")


if __name__ == "__main__":
    test_valid_chat_request()
    test_valid_chat_response_with_recommendations()
    test_empty_recommendations_for_clarification()
    test_invalid_test_type_raises_error()
    test_multi_code_test_type()
    test_too_many_recommendations_raises_error()
    print("\nAll schema tests complete.")
