from src.llm_facade.foo import foo


def test_foo():
    assert foo("foo") == "foo"
