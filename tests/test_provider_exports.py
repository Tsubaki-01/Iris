import iris.providers as providers


def test_provider_package_exports_only_active_provider_api() -> None:
    assert not hasattr(providers, "ProviderAdapter")
    assert not hasattr(providers, "OpenAIMessageAdapter")
    assert not hasattr(providers, "AnthropicMessageAdapter")
    assert hasattr(providers, "ProviderClient")
    assert hasattr(providers, "create_provider_client")
